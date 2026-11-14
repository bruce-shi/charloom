import { createContext, useContext } from "react";
export type Point3 = [number, number, number];
export type Stage =
  "body" | "assembled" | "fitted" | "parametric" | "aligned" | "bodyBaseline";
export type CharacterCase = {
  id: string;
  title: string;
  resultSha256: string;
  guides?: string;
  folder: string;
  report: string;
  hasFullBody: boolean;
  hasClothingShell: boolean;
  diagnosticOnly: boolean;
  sourceComparisonStatus?: string;
  sourceComparisonAutoPass?: boolean;
  diagnosticReason?: string;
  neckGeometryPassed: boolean;
  previousFolder?: string;
  previousHasFullBody?: boolean;
  bodyCenter: Point3;
  center: Point3;
  neckCenter: Point3;
  appearanceNote: string;
  note: string;
  seconds: number;
  newCrossings: number;
  landmarkRmse: Record<string, number>;
  /** MakeHuman transfer runs (`scripts/mh-ide-review.py`): the result
   * versions to choose from and the source scan in the result's frame.
   * Without them the head-fit stages above apply. */
  stages?: {
    id: string;
    label: string;
    file: string;
    /** Only these MakeHuman parts are shown (`charloom_part` mesh extras,
     * e.g. ["garment"] for the clothes-only view). */
    parts?: string[];
  }[];
  sourceFile?: string;
  suite?: string;
  /** Column-major source-to-result matrix applied to `sourceFile` when the
   * scan is referenced in place (scripts/mh-ide-sync.py). */
  sourceMatrix?: number[];
  /** Result runs from scripts/mh-ide-sync.py. */
  character?: string;
  alias?: string;
  tag?: string;
  round?: string;
  change?: string;
  headRoute?: string;
  timestamp?: string;
  runDir?: string;
  location?: string;
  thumbnail?: string | null;
  renders?: string[];
  metrics?: RunMetrics;
};
export type RunMetrics = {
  inverted?: number | null;
  edgeMax?: number | null;
  skinP95Mm?: number | null;
  eyeMm?: number | null;
  faceBumps?: number | null;
  newFolds?: number | null;
  walkPct?: number | null;
  seconds?: number | null;
};
export const StudioContext = createContext<{
  cases: CharacterCase[];
  error: string;
}>({ cases: [], error: "" });
export const useStudio = () => useContext(StudioContext);
