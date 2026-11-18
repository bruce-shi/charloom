import { expect, test } from "bun:test";
import { AnimationClock } from "../src/studio/Viewport";

test("paused time and clip survive model replacement", () => {
  const clock = new AnimationClock();
  const first = clock.registerModel({}, { walk: 2, face: 8, talk: 12 });
  clock.setClip("talk");
  clock.seek(4.75);
  first();
  const second = clock.registerModel({}, { walk: 2, face: 8, talk: 12 });
  expect(clock.clip).toBe("talk");
  expect(clock.time()).toBe(4.75);
  expect(clock.playing).toBe(false);
  expect(clock.enabled).toBe(true);
  second();
});

test("unmounted model durations and clip availability are removed", () => {
  const clock = new AnimationClock();
  const first = clock.registerModel({}, { walk: 3, face: 8, talk: 12 });
  const second = clock.registerModel({}, { walk: 2, face: 0, talk: 0 });
  clock.seek(1.25);
  first();
  expect(clock.time()).toBe(1.25);
  expect(clock.duration).toBe(2);
  expect(clock.has("talk")).toBe(false);
  second();
  expect(clock.duration).toBe(0);
  expect(clock.time()).toBe(1.25);
});

test("model replacement retains playing state", () => {
  const clock = new AnimationClock();
  const detach = clock.registerModel({}, { walk: 2, face: 0, talk: 0 });
  clock.seek(0.44);
  clock.play();
  detach();
  clock.registerModel({}, { walk: 2, face: 0, talk: 0 });
  expect(clock.playing).toBe(true);
  expect(clock.time()).toBeGreaterThanOrEqual(0.44);
  expect(clock.time()).toBeLessThan(0.65);
});

test("embedded clips share the playhead across layers and unregister with their models", () => {
  const clock = new AnimationClock();
  const running = "embedded:running";
  const old = clock.registerModel({}, { walk: 1, [running]: 0.7 });
  clock.setClip(running);
  clock.seek(0.26);
  const current = clock.registerModel({}, { walk: 1, [running]: 0.7, "embedded:wave": 2 });
  old();
  expect(clock.clip).toBe(running);
  expect(clock.duration).toBe(0.7);
  expect(clock.time()).toBe(0.26);
  expect(clock.embeddedClips()).toEqual([running, "embedded:wave"]);
  current();
  expect(clock.embeddedClips()).toEqual([]);
});
