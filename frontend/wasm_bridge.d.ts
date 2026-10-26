/* tslint:disable */
/* eslint-disable */

/**
 * Benchmark Sobel: time SIMD vs scalar path on a synthetic grayscale image.
 * Returns a JS object `{simd_us, scalar_us, speedup, pixels, width, height}`.
 * On native (not WASM), simd and scalar are identical.
 */
export function bench_sobel(width: number, height: number, iterations: number): any;

/**
 * Classify a PDF from bytes. Returns pdfType, pageCount, pagesNeedingOcr, confidence.
 */
export function classifyPdf(data: Uint8Array): any;

export function classify_file(bytes: Uint8Array): any;

/**
 * Per-degree rotation classification (v2): 5 doc traits + flip + trust +
 * deterministic `rotation_deg`, `rotation_confidence`, `needs_review`.
 */
export function classify_file_v2(bytes: Uint8Array): any;

/**
 * Extract text items with positions from a PDF memory buffer.
 */
export function extractTextWithPositions(data: Uint8Array): any;

/**
 * Load a CRF model for NER-based PII detection.
 *
 * The model is used by `scan_pii` to detect ADDRESS, NAME, ACCOUNT
 * via sequence labeling, ensemble-merged with rule-based scanners.
 *
 * Pass an empty string to clear the model.
 */
export function load_crf_model(json: string): void;

export function load_model(json: string): void;

/**
 * Load the per-degree rework model manifest (v3; format-compatible with v2).
 *
 * Expects the v3 `model.json` shape `{labels, tau, version, trees}` written by
 * `train_model_v2.py`. Also accepts a legacy flat tree array for
 * compatibility, but then the 7-label count and the default tau apply.
 */
export function load_model_v2(json: string): void;

/**
 * Manifest version of the loaded model (0 when unset or not loaded).
 */
export function model_version(): number;

/**
 * Scan text for PII/PCI (credit card numbers, SSNs, emails, phones).
 *
 * Returns a JavaScript array of hit objects:
 *   [{kind: "PAN"|"SSN"|"PHONE"|"EMAIL"|"CVV"|"EXPIRY"|"ROUTING"|"ACCOUNT"|"DOB"|"TRACK"|"PIN"|"PIN_BLOCK",
 *     text: "matched string", start: byte_offset, end: byte_offset}, ...]
 */
export function scan_pii(text: string): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly bench_sobel: (a: number, b: number, c: number) => [number, number, number];
    readonly classify_file: (a: number, b: number) => [number, number, number];
    readonly classify_file_v2: (a: number, b: number) => [number, number, number];
    readonly load_crf_model: (a: number, b: number) => [number, number];
    readonly load_model: (a: number, b: number) => [number, number];
    readonly load_model_v2: (a: number, b: number) => [number, number];
    readonly model_version: () => number;
    readonly scan_pii: (a: number, b: number) => [number, number, number];
    readonly classifyPdf: (a: number, b: number) => [number, number, number];
    readonly extractTextWithPositions: (a: number, b: number) => [number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
