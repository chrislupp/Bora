/*
 * The Kulfan (CST) shape functions, as in src/kulfan.rs: eight Bernstein
 * weights per surface, a leading-edge mode and a trailing-edge thickness.
 * Used only to draw figures.
 */

export type Kulfan = {
    upper: number[];
    lower: number[];
    leadingEdge: number;
    teThickness: number;
};

const N = 8;

const binomial = (n: number, k: number): number => {
    let c = 1;
    for (let i = 1; i <= k; i += 1) {
        c = (c * (n - k + i)) / i;
    }
    return c;
};

export const bernstein = (k: number, x: number): number =>
    binomial(N - 1, k) * x ** k * (1 - x) ** (N - 1 - k);

export const classFunction = (x: number): number => Math.sqrt(x) * (1 - x);

export const leadingEdgeMode = (x: number): number => x * Math.max(1 - x, 0) ** (N + 0.5);

/** Height of one surface at chordwise station `x`. */
export function surface(shape: Kulfan, side: "upper" | "lower", x: number): number {
    const weights = side === "upper" ? shape.upper : shape.lower;
    const sum = weights.reduce((total, w, k) => total + w * bernstein(k, x), 0);
    const te = ((side === "upper" ? 1 : -1) * x * shape.teThickness) / 2;
    return classFunction(x) * sum + te + shape.leadingEdge * leadingEdgeMode(x);
}

/** Cosine-spaced stations from the leading to the trailing edge. */
export const stations = (count: number): number[] =>
    Array.from({ length: count }, (_, i) => 0.5 * (1 - Math.cos((Math.PI * i) / (count - 1))));
