/** Impostor rendering mode: auto (threshold-based), always, or never */
export type ImpostorMode = "auto" | "always" | "never";

export interface ImpostorSettings {
  mode: ImpostorMode;
  threshold: number;
}

/** Default atom count threshold for auto-switching to impostor rendering */
export const DEFAULT_IMPOSTOR_SETTINGS: ImpostorSettings = {
  mode: "auto",
  threshold: 5000,
};
