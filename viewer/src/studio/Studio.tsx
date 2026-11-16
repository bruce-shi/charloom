import { lazy, Suspense, useEffect, useState } from "react";
import {
  BrowserRouter,
  Link,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
} from "react-router";
import { ResultsLibrary } from "./ResultsLibrary";
import { StudioContext, type CharacterCase } from "./data";
import "./studio.css";

const CharacterPage = lazy(() => import("./CharacterPage"));
/** Every mh-transfer result, as registered by scripts/mh-ide-sync.py. */
const RESULTS = "/files/out/review/ide/results.json";

/** Browse the results library and compare character exports. */
function StudioShell() {
  const [cases, setCases] = useState<CharacterCase[]>([]),
    [error, setError] = useState("");
  const location = useLocation();
  const id = location.pathname.split("/")[2];
  const selected = cases.find((c) => c.id === id);
  useEffect(() => {
    const c = new AbortController();
    fetch(RESULTS, { signal: c.signal, cache: "no-store" })
      .then(async (r) => {
        if (r.status === 401) {
          throw new Error("此浏览器的本地会话尚未建立或已失效。请打开本地 Gallery 启动时输出的完整链接。");
        }
        if (!r.ok) {
          const detail = await r.json().catch(() => undefined);
          throw new Error(detail?.error?.message ?? `结果目录读取失败（HTTP ${r.status}）`);
        }
        return r.json();
      })
      .then((d) => setCases(Array.isArray(d?.runs) ? d.runs : []))
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => c.abort();
  }, []);
  useEffect(() => {
    document.title = `${selected?.title ?? "角色"} · Charloom Studio`;
  }, [selected]);
  return (
    <StudioContext.Provider value={{ cases, error }}>
      <div className="studio-shell explorer-hidden preview-mode">
        <header className="studio-header">
          <Link className="studio-brand" to="/assets">
            <span className="brand-mark">C</span>CHARLOOM
            <span className="brand-studio">STUDIO</span>
          </Link>
          <div className="header-divider" />
          <span className="workspace-name">
            角色浏览
          </span>
          <div className="toolbar-spacer" />
          <span className="local-engine">
            <i />
            本地工作区
          </span>
        </header>
        <main className="studio-content">
          <Suspense fallback={<div className="studio-empty">正在打开…</div>}>
            <Outlet />
          </Suspense>
        </main>
        <footer className="studio-status">
          <span>
            <span className="status-dot" />
            {cases.length} 个结果
          </span>
          <span>Charloom / {selected?.id ?? "assets"}</span>
          <div className="toolbar-spacer" />
          <span>文件在本机处理</span>
        </footer>
      </div>
    </StudioContext.Provider>
  );
}

export default function Studio() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<StudioShell />}>
          <Route index element={<Navigate replace to="/assets" />} />
          <Route path="assets" element={<ResultsLibrary />} />
          <Route path="characters/:characterId" element={<CharacterPage />} />
          <Route path="characters/:characterId/*" element={<CharacterPage />} />
          <Route path="*" element={<Navigate replace to="/assets" />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
