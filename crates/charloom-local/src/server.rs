use crate::{
    API,
    model::*,
    store::{Store, atomic_json, valid_id},
};
use anyhow::{Result, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{
        DefaultBodyLimit, Path, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::{HeaderMap, Request as HttpRequest, StatusCode},
    middleware::{self, Next},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use futures_util::{SinkExt, StreamExt, stream};
use serde_json::{Value, json};
use std::{convert::Infallible, path::PathBuf, sync::Arc, time::Duration};
use tower_http::services::{ServeDir, ServeFile};

pub struct Config {
    pub data_dir: PathBuf,
    pub web_dir: PathBuf,
    pub port: u16,
    pub dev_origin: Option<String>,
}
#[derive(Clone)]
struct App {
    store: Arc<Store>,
    worker: Arc<crate::worker::NativeWorker>,
    info: ServiceInfo,
    dev_origin: Option<String>,
    shutdown: tokio::sync::watch::Sender<bool>,
}
fn failure_detail(error: impl std::fmt::Display) -> (StatusCode, Value) {
    let text = error.to_string();
    let (status, code) = if text.starts_with("revision_conflict:") {
        (StatusCode::CONFLICT, "revision_conflict")
    } else if text.contains("project_exists") {
        (StatusCode::CONFLICT, "project_exists")
    } else if text.contains("project_not_found") || text.contains("blob_not_found") {
        (StatusCode::NOT_FOUND, "not_found")
    } else {
        (StatusCode::BAD_REQUEST, "invalid_request")
    };
    let revision = text
        .strip_prefix("revision_conflict:")
        .and_then(|s| s.parse::<u64>().ok());
    (
        status,
        json!({"error":{"code":code,"message":text,"current_revision":revision}}),
    )
}
fn failure(error: impl std::fmt::Display) -> Response {
    let (status, detail) = failure_detail(error);
    (status, Json(detail)).into_response()
}

async fn guarded(
    State(app): State<App>,
    req: HttpRequest<axum::body::Body>,
    next: Next,
) -> Response {
    let host = req
        .headers()
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let port = app.info.url.rsplit(':').next().unwrap_or("");
    if host != format!("127.0.0.1:{port}") && host != format!("localhost:{port}") {
        return (StatusCode::FORBIDDEN, "Local host required").into_response();
    }
    if let Some(origin) = req.headers().get("origin").and_then(|v| v.to_str().ok())
        && origin != app.info.url
        && origin != format!("http://localhost:{port}")
        && app.dev_origin.as_deref() != Some(origin)
    {
        return (StatusCode::FORBIDDEN, "Same-origin local client required").into_response();
    }
    if req.method() != axum::http::Method::GET
        && req.method() != axum::http::Method::HEAD
        && req
            .headers()
            .get("x-charloom-protocol")
            .and_then(|h| h.to_str().ok())
            != Some("1")
    {
        return (StatusCode::FORBIDDEN, "Charloom protocol header required").into_response();
    }
    if req.uri().path().starts_with(API) || req.uri().path().starts_with("/files/") {
        let bearer = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        let cookie_name = format!("charloom_session_{port}=");
        let cookie = req
            .headers()
            .get("cookie")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| {
                v.split(';')
                    .map(str::trim)
                    .find_map(|v| v.strip_prefix(&cookie_name))
            });
        let token = app.info.auth_token.as_deref();
        // Native clients may authenticate the WebSocket handshake via a second
        // subprotocol. Browsers use the existing HttpOnly session cookie.
        let socket_token = if req.uri().path() == format!("{API}/ws") {
            req.headers()
                .get("sec-websocket-protocol")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| {
                    v.split(',')
                        .map(str::trim)
                        .find_map(|v| v.strip_prefix("charloom.auth."))
                })
        } else {
            None
        };
        if bearer != token && cookie != token && socket_token != token || token.is_none() {
            return (StatusCode::UNAUTHORIZED,Json(json!({"error":{"code":"local_session_required","message":"请通过 charloom ide 打开本地工作台"}}))).into_response();
        }
    }
    next.run(req).await
}
async fn health(State(app): State<App>) -> Json<ServiceInfo> {
    let mut info = app.info;
    info.auth_token = None;
    Json(info)
}
async fn session(State(app): State<App>) -> Response {
    let port = app.info.url.rsplit(':').next().unwrap();
    let cookie = format!(
        "charloom_session_{port}={}; HttpOnly; SameSite=Strict; Path=/",
        app.info.auth_token.as_deref().unwrap()
    );
    (
        [(axum::http::header::SET_COOKIE, cookie)],
        Json(json!({"connected":true})),
    )
        .into_response()
}
async fn schema_handler() -> Json<Value> {
    Json(crate::model::schema())
}
fn execute(app: App, request: Request) -> Result<Value> {
    let store = &app.store;
    Ok(match request {
        Request::List => serde_json::to_value(store.list()?)?,
        Request::Create {
            name,
            id,
            document,
            assets,
        } => serde_json::to_value(store.create_with_assets(name, id, document, assets)?)?,
        Request::Read { id, pointer } => {
            let p = store.read(&id)?;
            if let Some(pointer) = pointer {
                let value = p
                    .document
                    .pointer(&pointer)
                    .ok_or_else(|| anyhow::anyhow!("pointer_missing"))?;
                json!({"project_id":id,"revision":p.revision,"value":value})
            } else {
                serde_json::to_value(p)?
            }
        }
        Request::Edit {
            id,
            expected_revision,
            changes,
            assets,
            actor,
            request_id,
        } => serde_json::to_value(store.edit_assets(
            &id,
            expected_revision,
            changes,
            actor.unwrap_or_else(|| "api".into()),
            request_id.unwrap_or_else(crate::store::uid),
            assets,
        )?)?,
        Request::Resources {
            id,
            expected_revision,
            assets,
            actor,
        } => serde_json::to_value(store.resources(
            &id,
            expected_revision,
            assets,
            actor.unwrap_or_else(|| "api".into()),
        )?)?,
        Request::OpenPath {
            path,
            project,
            expected_revision,
            role,
        } => serde_json::to_value(store.open_path(&path, project, expected_revision, role)?)?,
        Request::Run {
            id,
            expected_revision,
            request,
            request_pointer,
            bindings,
            result_name,
        } => store.run(
            &id,
            expected_revision,
            request,
            request_pointer,
            bindings,
            result_name,
        )?,
        Request::Browse { path } => {
            let path = path.unwrap_or(std::env::current_dir()?).canonicalize()?;
            ensure!(path.is_dir(), "not_a_directory");
            let mut entries = vec![];
            for entry in std::fs::read_dir(&path)?.take(2000) {
                let e = entry?;
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with('.') {
                    continue;
                }
                let p = e.path();
                let is_dir = p.is_dir();
                let format = p
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if is_dir
                    || ["glb", "gltf", "obj", "fbx", "stl", "json", "geometry"]
                        .contains(&format.as_str())
                {
                    entries.push(json!({"name":name,"path":p,"is_dir":is_dir}));
                }
            }
            entries.sort_by_key(|v| {
                (
                    !v["is_dir"].as_bool().unwrap_or(false),
                    v["name"].as_str().unwrap_or("").to_lowercase(),
                )
            });
            json!({"path":path,"parent":path.parent(),"entries":entries})
        }
        Request::Task {
            task,
            project,
            expected_revision,
        } => app
            .worker
            .execute(store, task, project, expected_revision)?,
        Request::Preview { id, tool } => {
            ensure!(
                app.info.web_dir.join("index.html").is_file(),
                "web_ide_not_built: run bun run build in viewer"
            );
            if let Some(id) = &id {
                store.read(id)?;
            }
            let mut url = app.info.url.clone();
            if let Some(id) = id {
                url.push_str(&format!("/workspace/{id}"));
            }
            if let Some(tool) = tool {
                ensure!(
                    [
                        "prepare", "object", "body", "guide", "fit", "export", "sculpt", "remesh"
                    ]
                    .contains(&tool.as_str()),
                    "invalid_preview_tool"
                );
                url.push_str(&format!("?tool={tool}"));
            }
            url.push_str(&format!(
                "#auth={}",
                app.info.auth_token.as_deref().unwrap()
            ));
            json!({"url":url,"data_dir":app.info.data_dir,"project_storage":"local_service"})
        }
    })
}
async fn action(
    State(app): State<App>,
    request: Result<Json<Request>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let request = match request {
        Ok(Json(request)) => request,
        Err(e) => return failure(e.body_text()),
    };
    match tokio::task::spawn_blocking(move || execute(app, request)).await {
        Ok(Ok(value)) => ([("cache-control", "no-store")], Json(value)).into_response(),
        Ok(Err(e)) => failure(e),
        Err(e) => failure(e),
    }
}

async fn upload(State(app): State<App>, bytes: Bytes) -> Response {
    match tokio::task::spawn_blocking(move || app.store.blob(&bytes)).await {
        Ok(Ok(hash)) => Json(json!({"blob":hash})).into_response(),
        Ok(Err(e)) => failure(e),
        Err(e) => failure(e),
    }
}
async fn blob(State(app): State<App>, Path(hash): Path<String>) -> Response {
    match app.store.blob_path(&hash) {
        Ok(path) => match tokio::fs::read(path).await {
            Ok(bytes) => (
                [
                    ("content-type", "application/octet-stream"),
                    ("cache-control", "public, max-age=31536000, immutable"),
                ],
                bytes,
            )
                .into_response(),
            Err(e) => failure(e),
        },
        Err(e) => failure(e),
    }
}
async fn result(State(app): State<App>, Path((id, result)): Path<(String, String)>) -> Response {
    if !valid_id(&id) || !valid_id(&result) {
        return failure("invalid_result_id");
    }
    match tokio::fs::read(
        app.store
            .root
            .join("results")
            .join(id)
            .join(format!("{result}.json")),
    )
    .await
    {
        Ok(bytes) => (
            [
                ("content-type", "application/json"),
                ("cache-control", "no-store"),
            ],
            bytes,
        )
            .into_response(),
        Err(e) => failure(e),
    }
}
async fn artifact(State(app): State<App>, Path(relative): Path<String>) -> Response {
    let lookup = (|| -> Result<PathBuf> {
        let (base, tail) = if let Some(local) = relative.strip_prefix("local/jobs/") {
            (app.store.root.join("jobs"), local)
        } else if relative.starts_with("out/") || relative.starts_with("fixtures/") {
            (app.worker.source_root.clone(), relative.as_str())
        } else {
            anyhow::bail!("artifact_forbidden")
        };
        let base = base.canonicalize()?;
        let file = base.join(tail).canonicalize()?;
        ensure!(
            file.starts_with(&base) && file.is_file(),
            "artifact_forbidden"
        );
        ensure!(
            ["glb", "json", "png", "jpg", "jpeg", "md"]
                .contains(&file.extension().and_then(|s| s.to_str()).unwrap_or("")),
            "artifact_format_forbidden"
        );
        Ok(file)
    })();
    match lookup {
        Ok(path) => {
            let content = match path.extension().and_then(|s| s.to_str()).unwrap_or("") {
                "glb" => "model/gltf-binary",
                "json" => "application/json",
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                _ => "text/plain; charset=utf-8",
            };
            match tokio::fs::read(path).await {
                Ok(bytes) => (
                    [("content-type", content), ("cache-control", "no-store")],
                    bytes,
                )
                    .into_response(),
                Err(e) => failure(e),
            }
        }
        Err(e) => failure(e),
    }
}
async fn events(
    State(app): State<App>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let receiver = app.store.events.subscribe();
    let shutdown = app.shutdown.subscribe();
    let stream = stream::unfold(
        (receiver, shutdown, true),
        |(mut receiver, mut shutdown, first)| async move {
            if first {
                return Some((
                    Ok(Event::default().event("resync").data("{}")),
                    (receiver, shutdown, false),
                ));
            }
            tokio::select! {
                _=shutdown.changed()=>None,
                event=receiver.recv()=>match event {
                    Ok(value)=>{let event=match value {Notice::Project(receipt)=>Event::default().event("project_changed").json_data(receipt).unwrap(),Notice::Library(library)=>Event::default().event("library_changed").json_data(json!({"library":library})).unwrap()};Some((Ok(event),(receiver,shutdown,false)))},
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_))=>Some((Ok(Event::default().event("resync").data("{}")),(receiver,shutdown,false))),
                    Err(_)=>None,
                }
            }
        },
    );
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
}
async fn websocket(
    State(app): State<App>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let version = headers
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|p| p.trim() == "charloom.v1"));
    if !version {
        return failure("websocket_protocol_required: charloom.v1");
    }
    upgrade
        .protocols(["charloom.v1"])
        .max_message_size(2 * 1024 * 1024)
        .on_upgrade(move |socket| socket_session(socket, app))
}
fn socket_notice(notice: Notice) -> SocketServer {
    match notice {
        Notice::Project(receipt) => SocketServer::ProjectChanged { receipt },
        Notice::Library(library) => SocketServer::LibraryChanged { library },
    }
}
async fn socket_session(socket: WebSocket, app: App) {
    let (mut writer, mut reader) = socket.split();
    let mut notices = app.store.events.subscribe();
    let mut shutdown = app.shutdown.subscribe();
    let (responses, mut incoming) = tokio::sync::mpsc::channel::<SocketServer>(16);
    let slots = Arc::new(tokio::sync::Semaphore::new(8));
    let encode = |frame: &SocketServer| {
        Message::Text(
            serde_json::to_string(frame)
                .expect("socket protocol JSON")
                .into(),
        )
    };
    if writer
        .send(encode(&SocketServer::Ready { protocol: 1 }))
        .await
        .is_err()
    {
        return;
    }
    if writer.send(encode(&SocketServer::Resync)).await.is_err() {
        return;
    }
    let mut heartbeat = tokio::time::interval(Duration::from_secs(15));
    loop {
        let outgoing = tokio::select! {
            _=shutdown.changed()=>break,
            _=heartbeat.tick()=>{if writer.send(Message::Ping(Bytes::new())).await.is_err(){break;}continue;},
            reply=incoming.recv()=>match reply{Some(reply)=>reply,None=>break},
            notice=notices.recv()=>match notice{Ok(notice)=>socket_notice(notice),Err(tokio::sync::broadcast::error::RecvError::Lagged(_))=>SocketServer::Resync,Err(_)=>break},
            message=reader.next()=>{
                match message {
                    Some(Ok(Message::Text(text)))=>{
                        match serde_json::from_str::<SocketClient>(text.as_str()) {
                            Ok(SocketClient::Action{id,request}) if valid_id(&id)=>{
                                match slots.clone().try_acquire_owned() {
                                    Ok(permit)=>{let app=app.clone();let responses=responses.clone();tokio::spawn(async move{
                                        let result=tokio::task::spawn_blocking(move||execute(app,request)).await;
                                        let frame=match result {Ok(Ok(value))=>SocketServer::Result{id,value},Ok(Err(e))=>{let(status,detail)=failure_detail(e);SocketServer::Error{id:Some(id),status:status.as_u16(),detail}},Err(e)=>{let(status,detail)=failure_detail(e);SocketServer::Error{id:Some(id),status:status.as_u16(),detail}}};
                                        let _=responses.send(frame).await;drop(permit);
                                    });continue;},
                                    Err(_)=>SocketServer::Error{id:Some(id),status:429,detail:json!({"error":{"code":"socket_busy","message":"Too many pending actions"}})},
                                }
                            },
                            Ok(_)=>SocketServer::Error{id:None,status:400,detail:json!({"error":{"code":"invalid_request_id","message":"Invalid socket request ID"}})},
                            Err(e)=>SocketServer::Error{id:None,status:400,detail:json!({"error":{"code":"invalid_socket_action","message":e.to_string()}})},
                        }
                    },
                    Some(Ok(Message::Ping(bytes)))=>{if writer.send(Message::Pong(bytes)).await.is_err(){break;}continue;},
                    Some(Ok(Message::Pong(_)))=>continue,
                    Some(Ok(Message::Close(_)))|None|Some(Err(_))=>break,
                    Some(Ok(Message::Binary(_)))=>SocketServer::Error{id:None,status:400,detail:json!({"error":{"code":"text_actions_required","message":"Use HTTP for binary assets"}})},
                }
            },
        };
        if writer.send(encode(&outgoing)).await.is_err() {
            break;
        }
    }
    let _ = writer.send(Message::Close(None)).await;
}
pub async fn serve(config: Config) -> Result<()> {
    serve_until(config, async {
        #[cfg(unix)]
        {
            let mut term =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("SIGTERM handler");
            tokio::select! {_=tokio::signal::ctrl_c()=>{},_=term.recv()=>{}}
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
    })
    .await
}
pub async fn serve_until(
    config: Config,
    stop: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<()> {
    if let Some(origin) = &config.dev_origin {
        crate::client::check_url(origin)?;
    }
    let store = Arc::new(Store::new(&config.data_dir)?);
    let listener =
        tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, config.port)).await?;
    let info = ServiceInfo {
        protocol: 1,
        pid: std::process::id(),
        url: format!("http://127.0.0.1:{}", listener.local_addr()?.port()),
        data_dir: store.root.clone(),
        web_dir: config.web_dir.clone(),
        auth_token: Some(crate::store::uid()),
    };
    let (shutdown, _) = tokio::sync::watch::channel(false);
    let app = App {
        store: store.clone(),
        worker: Arc::new(crate::worker::NativeWorker::new(&store.root)),
        info: info.clone(),
        dev_origin: config.dev_origin,
        shutdown: shutdown.clone(),
    };
    let router = Router::new()
        .route("/files/{*relative}", get(artifact))
        .route(&format!("{API}/session"), post(session))
        .route(&format!("{API}/health"), get(health))
        .route(&format!("{API}/schema"), get(schema_handler))
        .route(&format!("{API}/action"), post(action))
        .route(&format!("{API}/blobs"), post(upload))
        .route(&format!("{API}/blobs/{{hash}}"), get(blob))
        .route(&format!("{API}/events"), get(events))
        .route(&format!("{API}/ws"), get(websocket))
        .route(&format!("{API}/results/{{id}}/{{result}}"), get(result))
        .fallback_service(
            ServeDir::new(&config.web_dir)
                .fallback(ServeFile::new(config.web_dir.join("index.html"))),
        )
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(app.clone(), guarded))
        .with_state(app);
    atomic_json(&store.root.join("server.json"), &info)?;
    let mut public = info.clone();
    public.auth_token = None;
    println!("{}", serde_json::to_string(&public)?);
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            stop.await;
            let _ = shutdown.send(true);
        })
        .await?;
    let _ = std::fs::remove_file(store.root.join("server.json"));
    Ok(())
}
