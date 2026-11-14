import { createRoot } from "react-dom/client";
import "./style.css";
import Studio from "./studio/Studio";
import { localClient } from "./local/client";
// Establish the local session before catalog assets or project data are read.
void localClient
  .connect()
  .catch(() => {})
  .then(() => {
    createRoot(document.getElementById("root")!).render(<Studio />);
  });
