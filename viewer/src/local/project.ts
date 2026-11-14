/** A project document of the local service (`charloom serve`). The studio
 * only opens a session with it; project contents are not edited here. */
export type ProjectManifest = {
  format: "charloom-project";
  version: 1;
  id: string;
  name: string;
  [key: string]: unknown;
};
