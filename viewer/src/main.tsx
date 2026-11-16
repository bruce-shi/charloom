import { createRoot } from "react-dom/client";
import "./style.css";
import Studio from "./studio/Studio";

async function openGallery() {
  const auth = new URLSearchParams(location.hash.slice(1)).get("auth");
  if (auth) {
    const session = await fetch("/api/local/v1/session", {
      method: "POST", headers: { authorization: `Bearer ${auth}`, "x-charloom-protocol": "1" },
    });
    if (session.ok) history.replaceState(history.state, "", location.pathname + location.search);
  }
}
void openGallery().finally(() => createRoot(document.getElementById("root")!).render(<Studio />));
