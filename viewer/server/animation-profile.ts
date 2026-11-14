export type InspectedJoint = {
  index: number;
  name: string;
  parent: number | null;
};
export type NativeMapping = {
  mappings: { source: number; target: number }[];
  target_deform_coverage: number;
};
// Keep the adapter's exception narrow: MPFB adds face and split limb bones that
// have no counterpart in Mixamo. Every major body joint must still be mapped.
export function retargetSettings(
  joints: InspectedJoint[],
  mapping: NativeMapping,
) {
  const normalized = (name: string) =>
    name.replace(/[^a-z0-9]/gi, "").toLowerCase();
  const byName = new Map(joints.map((j) => [normalized(j.name), j])),
    mapped = new Set(mapping.mappings.map((m) => m.target));
  const sides = ["L", "R"];
  const mpfb = [
    "root",
    "spine05",
    "spine01",
    ...sides.flatMap((s) =>
      ["upperarm01", "lowerarm02", "wrist", "upperleg01", "foot"].map(
        (n) => `${n}.${s}`,
      ),
    ),
  ].every((n) => byName.has(normalized(n)));
  if (mpfb) {
    const required = [
      "root",
      "spine05",
      "spine03",
      "spine01",
      "neck01",
      "head",
      ...sides.flatMap((s) =>
        [
          "clavicle",
          "upperarm01",
          "lowerarm01",
          "wrist",
          "upperleg01",
          "lowerleg01",
          "foot",
        ].map((n) => `${n}.${s}`),
      ),
    ];
    const missing = required.filter(
      (n) => !mapped.has(byName.get(normalized(n))?.index ?? -1),
    );
    if (missing.length)
      throw new Error(`动作缺少角色所需的骨骼：${missing.join("、")}`);
  }
  const compact = (name: string) =>
    name
      .split(/[:|]/)
      .at(-1)!
      .replace(/[^a-z0-9]/gi, "")
      .toLowerCase()
      .replace(/^hc/, "");
  const directWrists = joints
    .filter(
      (j) =>
        /^(?:left|right)hand$|^hand[lr]$/.test(compact(j.name)) &&
        mapped.has(j.index),
    )
    .filter((j) => {
      const parent = joints.find((p) => p.index === j.parent);
      return (
        parent &&
        mapped.has(parent.index) &&
        /^(?:left|right)(?:forearm|lowerarm)$|^(?:forearm|lowerarm)[lr]$/.test(
          compact(parent.name),
        )
      );
    });
  return {
    min_target_coverage: mpfb ? mapping.target_deform_coverage : 0.5,
    distribute_wrist_twist: !mpfb && directWrists.length > 0,
  };
}
