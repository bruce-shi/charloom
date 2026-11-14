import { useEffect, useRef, useState, type ReactNode } from "react";
import { X } from "lucide-react";

const TABS = ["renders", "shape"] as const;
const IMAGE_GROUPS = ["参考", "近景", "渲染"] as const;

 
function imageLabel(src: string) {
  const parts = decodeURIComponent(src.split(/[?#]/)[0]).split("/");
  const name = parts.pop()!.replace(/\.(png|jpe?g|webp)$/i, "");
  const parent = parts.pop();
  const label = name
    .replace(/(^|-)eevee-/, "$1")
    .replace(/^qa-front$/, "参考 · 正面")
    .replace(/^qa-side$/, "参考 · 侧面")
    .replace(/^4-garments-character-/, "角色 · ")
    .replace(/^3-skin-body-/, "身体 · ")
    .replace(/^eyes-/, "眼部 · ")
    .replace(/-rest-crop$/, " · 静止特写")
    .replace(/-rest$/, " · 静止");
  return parent && parent !== "renders" && parent !== "0-guide"
    ? `${parent}/${label}`
    : label;
}

function imageGroup(src: string): (typeof IMAGE_GROUPS)[number] {
  const label = decodeURIComponent(src.split(/[?#]/)[0]);
  if (/\/0-guide\/|\/(qa-|evidence-)/i.test(label)) return "参考";
  if (/\/(eyes-|neck[/-]|collar-)/i.test(label)) return "近景";
  return "渲染";
}

/** Keyed by the selected run; body-shape values remain in CharacterPage. */
export function CharacterSidebar({
  renders = [],
  children,
}: {
  renders?: string[];
  children: ReactNode;
}) {
  const [tab, setTab] = useState<(typeof TABS)[number]>(
    renders.length ? "renders" : "shape",
  );
  const [zoomed, setZoomed] = useState<string>();
  return (
    <aside className="character-sidebar" aria-label="角色参考与体型">
      <div
        className="character-sidebar-tabs"
        role="tablist"
        aria-label="侧栏内容"
        onKeyDown={(event) => {
          let next: typeof tab;
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            next = tab === "renders" ? "shape" : "renders";
          } else if (event.key === "Home") next = "renders";
          else if (event.key === "End") next = "shape";
          else return;
          event.preventDefault();
          setTab(next);
          event.currentTarget.querySelector<HTMLButtonElement>(`#${next}-tab`)?.focus();
        }}
      >
        {TABS.map((name) => (
          <button
            key={name}
            id={`${name}-tab`}
            role="tab"
            aria-selected={tab === name}
            aria-controls={`${name}-panel`}
            tabIndex={tab === name ? 0 : -1}
            onClick={() => setTab(name)}
          >
            {name === "renders" ? `参考图 ${renders.length}` : "MakeHuman 体型"}
          </button>
        ))}
      </div>
      <div
        id="renders-panel"
        className="character-sidebar-panel reference-panel"
        role="tabpanel"
        aria-labelledby="renders-tab"
        hidden={tab !== "renders"}
        tabIndex={0}
      >
        {!renders.length && <p className="shape-panel-empty">这个结果暂无参考图。</p>}
        {IMAGE_GROUPS.map((group) => {
          const images = renders.filter((src) => imageGroup(src) === group);
          if (!images.length) return null;
          return (
            <section className="render-group" key={group} aria-label={group}>
              <h3>{group}</h3>
              <div className="render-gallery">
                {images.map((src) => (
                  <button key={src} onClick={() => setZoomed(src)} title={imageLabel(src)}>
                    <img src={src} alt={imageLabel(src)} loading="lazy" />
                    <span>{imageLabel(src)}</span>
                  </button>
                ))}
              </div>
            </section>
          );
        })}
      </div>
      <div
        id="shape-panel"
        className="character-sidebar-panel"
        role="tabpanel"
        aria-labelledby="shape-tab"
        hidden={tab !== "shape"}
        tabIndex={0}
      >
        {children}
      </div>
      {zoomed && <RenderZoom src={zoomed} onClose={() => setZoomed(undefined)} />}
    </aside>
  );
}

function RenderZoom({ src, onClose }: { src: string; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current!;
    element.showModal();
    return () => element.close();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="render-zoom"
      aria-label={imageLabel(src)}
      onClick={onClose}
      onClose={onClose}
    >
      <img src={src} alt={imageLabel(src)} />
      <button className="tool-button" aria-label="关闭大图">
        <span>{imageLabel(src)}</span>
        <X size={14} />
      </button>
    </dialog>
  );
}
