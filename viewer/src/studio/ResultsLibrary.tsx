import { useMemo, useState } from "react";
import { Link } from "react-router";
import { ArrowUpRight, Search } from "lucide-react";
import { useStudio, type CharacterCase } from "./data";
import {
  CARD_METRICS,
  characterOf,
  metricText,
  metricWarn,
  resultRuns,
  shortTime,
} from "./results";

/** Preview mode library: every mh-transfer result, grouped by character
 * (most recently updated character first), newest run first. */
export function ResultsLibrary() {
  const { cases, error } = useStudio();
  const [query, setQuery] = useState("");
  const runs = useMemo(() => resultRuns(cases), [cases]);
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const shown = runs.filter((c) => {
    const text = [
      characterOf(c),
      c.alias,
      c.tag,
      c.round,
      c.change,
      c.headRoute,
      c.id,
    ]
      .join(" ")
      .toLowerCase();
    return words.every((w) => text.includes(w));
  });
  const groups = new Map<string, CharacterCase[]>();
  for (const c of shown) {
    const key = characterOf(c);
    groups.set(key, [...(groups.get(key) ?? []), c]);
  }
  const characters = new Set(runs.map(characterOf)).size;
  return (
    <div className="studio-page results-library">
      <div className="page-title">
        <div>
          <h1>
            迁移结果 <span>只读预览</span>
          </h1>
          <p>
            左侧原扫描、右侧结果；在角色内可切换或对比该角色的其他结果。由
            scripts/mh-ide-sync.py 登记。
          </p>
        </div>
        <span className="count-badge">
          {runs.length} 个结果 · {characters} 个角色
        </span>
      </div>
      <label className="asset-search">
        <Search size={16} />
        <input
          aria-label="筛选结果"
          placeholder="筛选：角色、标签（eyes2、collar3）、改动…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      {!cases.length && !error && <p role="status">正在读取结果…</p>}
      {!!cases.length && !runs.length && (
        <p className="asset-empty">
          还没有登记的结果。运行 python3 scripts/mh-ide-sync.py 后刷新。
        </p>
      )}
      {!!runs.length && !shown.length && (
        <p className="asset-empty" role="status">
          没有匹配的结果。
        </p>
      )}
      {[...groups].map(([character, list]) => (
        <section
          key={character}
          className="asset-section"
          aria-label={character}
        >
          <div className="asset-section-heading">
            <h2>{character}</h2>
            <span>
              {list.length} 个结果 · 最新 {shortTime(list[0].timestamp)}
            </span>
          </div>
          <div className="asset-cards result-cards">
            {list.map((c) => (
              <ResultCard key={c.id} run={c} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function ResultCard({ run }: { run: CharacterCase }) {
  return (
    <Link
      className="asset-card result-card"
      to={`/characters/${encodeURIComponent(run.id)}`}
      title={run.appearanceNote}
    >
      <div className="asset-card-art">
        {run.thumbnail ? (
          <img src={run.thumbnail} alt="" loading="lazy" />
        ) : (
          <svg width="70" height="92" viewBox="0 0 100 130" aria-hidden="true">
            <circle cx="50" cy="18" r="10" />
            <path d="M50 28V77M26 38L50 33L74 38M26 38L17 64L10 88M74 38L83 64L90 88M50 77L34 78L30 115M50 77L66 78L70 115" />
          </svg>
        )}
        <span className="asset-stage">{run.tag}</span>
        {run.round && <span className="asset-index">{run.round}</span>}
      </div>
      <div className="asset-card-body">
        <h2>
          {run.change || run.tag}
          <ArrowUpRight size={15} />
        </h2>
        <span className="result-when">
          {shortTime(run.timestamp)} · 头部 {run.headRoute}
        </span>
        <dl className="result-card-metrics">
          {CARD_METRICS.map((m) => (
            <div key={m.key} className={metricWarn(run.metrics, m) ? "amber" : ""}>
              <dt>{m.label}</dt>
              <dd>
                {metricText(run.metrics, m)}
                <small>{m.unit}</small>
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </Link>
  );
}
