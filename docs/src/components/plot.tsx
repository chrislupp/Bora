import { useId, type ReactNode } from "react";

/*
 * Shared pieces of the documentation's figures. Every chart is drawn on the
 * ruled graph sheet, and its scales are chosen so that the sheet's rulings
 * are the chart's grid: one small square is U units, five make a heavy line.
 * Lettering is set with its baseline just above a ruling, so that it sits
 * inside one row of small squares and no ruling strikes through it.
 * Series are told apart by open symbols and states by line type. Styles live
 * in src/css/custom.css under `.bora-plot`.
 */

/** Side of one small square of the sheet, in SVG units. */
export const U = 12;

export type Point = [number, number];
export type Scale = (value: number) => number;

export const linear =
    (d0: number, d1: number, r0: number, r1: number): Scale =>
    (value) =>
        r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);

export function pathFrom(points: Point[]): string {
    return points
        .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`)
        .join(" ");
}

/** Typographic minus sign for negative numbers in labels. */
export const minus = (text: string): string => text.replace(/-/g, "−");

/** The ruling of the graph sheet, filling a `width` by `height` area. */
export function Sheet({
    width,
    height,
}: {
    width: number | string;
    height: number | string;
}): ReactNode {
    const id = `sheet${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
    return (
        <g aria-hidden="true">
            <defs>
                <pattern id={`${id}a`} width={U} height={U} patternUnits="userSpaceOnUse">
                    <path className="sheet-minor" d={`M${U} 0.5 H0.5 V${U}`} />
                </pattern>
                <pattern id={`${id}b`} width={5 * U} height={5 * U} patternUnits="userSpaceOnUse">
                    <path className="sheet-major" d={`M${5 * U} 0.6 H0.6 V${5 * U}`} />
                </pattern>
            </defs>
            <rect width={width} height={height} fill={`url(#${id}a)`} />
            <rect width={width} height={height} fill={`url(#${id}b)`} />
        </g>
    );
}

export type SymbolKind = "circle" | "triangle" | "square";
export const SYMBOLS: SymbolKind[] = ["circle", "triangle", "square"];

export function PlotSymbol({
    kind,
    x,
    y,
    r = 4.4,
    className = "symbol",
}: {
    kind: SymbolKind;
    x: number;
    y: number;
    r?: number;
    className?: string;
}): ReactNode {
    if (kind === "circle") {
        return <circle className={className} cx={x} cy={y} r={r} />;
    }
    if (kind === "triangle") {
        const d = `M${x} ${y - r - 1} L${x + r + 0.6} ${y + r - 0.6} L${x - r - 0.6} ${y + r - 0.6} Z`;
        return <path className={className} d={d} />;
    }
    const side = 2 * r - 0.8;
    return (
        <rect className={className} x={x - side / 2} y={y - side / 2} width={side} height={side} />
    );
}

export type Box = { left: number; right: number; top: number; bottom: number };

/**
 * Horizontal room the y-axis title needs left of the box: the numerals end
 * 10 units left of the axis, and the title is set rotated beyond them.
 */
export const Y_TITLE_ROOM = 5 * U;

/** The left and bottom axes on the sheet, numbered at the given values and titled. */
export function Chart({
    box,
    x,
    y,
    xTicks,
    yTicks,
    xFormat = String,
    yFormat = String,
    xTitle,
    yTitle,
}: {
    box: Box;
    x: Scale;
    y: Scale;
    xTicks: number[];
    yTicks: number[];
    xFormat?: (value: number) => string;
    yFormat?: (value: number) => string;
    xTitle?: ReactNode;
    yTitle?: ReactNode;
}): ReactNode {
    const yTitleX = box.left - Y_TITLE_ROOM + 4;
    const yTitleY = (box.top + box.bottom) / 2;
    return (
        <g>
            <path
                className="frame"
                d={`M${box.left} ${box.top} V${box.bottom} H${box.right}`}
            />
            {xTicks.map((t) => (
                <g key={`x${t}`}>
                    <path className="tick" d={`M${x(t).toFixed(1)} ${box.bottom} v6`} />
                    <text x={x(t)} y={box.bottom + 2 * U - 2} textAnchor="middle">
                        {minus(xFormat(t))}
                    </text>
                </g>
            ))}
            {yTicks.map((t) => (
                <g key={`y${t}`}>
                    <path className="tick" d={`M${box.left} ${y(t).toFixed(1)} h-6`} />
                    <text x={box.left - 10} y={y(t) + 5} textAnchor="end">
                        {minus(yFormat(t))}
                    </text>
                </g>
            ))}
            {xTitle && (
                <text x={(box.left + box.right) / 2} y={box.bottom + 4 * U - 2} textAnchor="middle">
                    {xTitle}
                </text>
            )}
            {yTitle && (
                <text
                    x={yTitleX}
                    y={yTitleY}
                    textAnchor="middle"
                    transform={`rotate(-90 ${yTitleX} ${yTitleY})`}
                >
                    {yTitle}
                </text>
            )}
        </g>
    );
}
