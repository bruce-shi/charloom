import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
const folder=fs.mkdtempSync(path.join(os.tmpdir(),'charloom-motion-chrome-'));
const output=process.env.CHARLOOM_HEAD_DIR || 'out/review/kawaii-head-fit';
fs.mkdirSync(output,{recursive:true});
const galleryUrl = process.env.CHARLOOM_GALLERY_URL;
if (!galleryUrl) throw Error('Set CHARLOOM_GALLERY_URL to the local Gallery session link');
const gallery = new URL(galleryUrl);
const token = new URLSearchParams(gallery.hash.slice(1)).get('auth');
if (!token || !['localhost', '127.0.0.1'].includes(gallery.hostname)) throw Error('Expected a local Gallery session link');
const port = gallery.port;
const proc=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--remote-debugging-port=0','--user-data-dir='+folder,'--no-first-run','--no-default-browser-check','--disable-extensions','--window-size=1280,940','--enable-unsafe-swiftshader','about:blank'],{stdio:'ignore'});
const sleep=ms=>new Promise(done=>setTimeout(done,ms));
const wait=async(fn,ms=60000)=>{const start=Date.now();while(Date.now()-start<ms){let v=await fn();if(v)return v;await sleep(250);}throw Error('wait timeout');};
let ws;
try{
const endpoint=folder+'/DevToolsActivePort';
await wait(()=>fs.existsSync(endpoint));const cdp='http://127.0.0.1:'+fs.readFileSync(endpoint,'utf8').split('\n')[0];
const targets=await(await fetch(cdp+'/json/list')).json();ws=new WebSocket(targets.find(x=>x.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));const pending=new Map();let id=0;
let loadedUrls=[];ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Network.responseReceived')loadedUrls.push([m.params.response.url,m.params.response.status]);if(m.id){const pair=pending.get(m.id);if(pair){pending.delete(m.id);if(m.error)pair[1](Error(m.error.message));else pair[0](m.result);}}};
const rpc=(method,params={})=>new Promise((yes,no)=>{pending.set(++id,[yes,no]);ws.send(JSON.stringify({id,method,params}));});
const js=async(expression)=>{let z=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(z.exceptionDetails)throw Error(z.exceptionDetails.exception?.description||z.exceptionDetails.text);return z.result.value;};
await rpc('Page.enable');await rpc('Runtime.enable');await rpc('Network.enable');
const auth=await rpc('Network.setCookie',{url:gallery.origin,name:'charloom_session_'+port,value:token,httpOnly:true,sameSite:'Strict'});if(!auth.success)throw Error('local session not set');

const cat=JSON.parse(fs.readFileSync('out/review/ide/results.json','utf8'));
const stages=(process.env.STAGES||'body,character').split(',');
for (const name of process.argv.slice(2)) for(const stage of stages) {
 const entry=cat.runs.find(row=>row.alias===name);
 const id=entry.id;
 await rpc('Page.navigate',{url:gallery.origin+'/characters/'+id+'?stage='+stage});
 await wait(()=>js("!!document.querySelector('button[aria-label=播放动画]:enabled') && document.title.length>0"));
 const click=async(t)=>js("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==="+JSON.stringify(t)+").click()");
 await click('走路');await sleep(150);
 await js("document.querySelector('button[aria-label=播放动画]').click()");await sleep(450);const first=await js("document.querySelector('.walk-time').textContent");await sleep(450);const last=await js("document.querySelector('.walk-time').textContent");if(first===last)throw Error('walk clock stalled');
 await js("document.querySelector('button[aria-label=暂停动画]').click()");
 await click('静止姿势');await sleep(400);
 const full=await rpc('Page.captureScreenshot',{format:'png',fromSurface:true});fs.writeFileSync(output+'/'+name+'-'+stage+'-full.png',Buffer.from(full.data,'base64'));
 await click('说话');await js("document.querySelector('button[aria-label=播放动画]').click()");await sleep(400);await js("document.querySelector('button[aria-label=暂停动画]').click()");
 await click('表情');await click('面部 3/4');await sleep(450);
 for(const [key,t] of [['quarter','中性'],['quarter-blink','眨眼']]){
  await click(t);await sleep(600);
  const r=await rpc('Page.captureScreenshot',{format:'png',fromSurface:true});
  fs.writeFileSync(output+'/'+name+'-'+stage+'-'+key+'.png',Buffer.from(r.data,'base64'));
 }
 await click('面部');
 for(const [key,label] of [['neutral','中性'],['blink','眨眼'],['glance','向左看'],['mouth','张嘴'],['up','向上看']]){
  await click(label);await sleep(600);
  const frame=await rpc('Page.captureScreenshot',{format:'png',fromSurface:true});
  fs.writeFileSync(path.join(output,name+'-'+stage+'-'+key+'.png'),Buffer.from(frame.data,'base64'));
 }
 if(!loadedUrls.some(([url,status])=>new URL(url).pathname===entry.stages.find(s=>s.id===stage).file && status===200))throw Error('Stage failed to load: '+name+' '+stage);
 console.log(name,'captured');
}
}finally{if(ws)ws.close();proc.kill('SIGTERM');}
