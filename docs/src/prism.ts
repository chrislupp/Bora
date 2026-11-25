import type { PrismTheme } from "prism-react-renderer";

/*
 * Code listings are highlighted in muted inks, the colours of drafting pens
 * rather than a screen palette: blue keywords (also bold), green strings, wine
 * numbers and constants, violet functions, teal types, grey comments (also
 * italic) and punctuation. The sheet's orange is kept out of code, as it is
 * out of all text. Every ink meets 4.5:1 against its paper.
 */
type Inks = {
    ink: string;
    paper: string;
    comment: string;
    keyword: string;
    string: string;
    number: string;
    func: string;
    type: string;
    punctuation: string;
};

function listing(c: Inks): PrismTheme {
    return {
        plain: { color: c.ink, backgroundColor: c.paper },
        styles: [
            {
                types: ["comment", "prolog", "doctype", "cdata"],
                style: { color: c.comment, fontStyle: "italic" },
            },
            {
                types: ["keyword", "atrule", "important", "tag", "selector", "rule"],
                style: { color: c.keyword, fontWeight: "bold" },
            },
            {
                types: ["string", "char", "attr-value", "url", "regex", "inserted"],
                style: { color: c.string },
            },
            {
                types: ["number", "boolean", "constant", "symbol", "entity", "deleted"],
                style: { color: c.number },
            },
            {
                types: ["function", "function-name", "macro"],
                style: { color: c.func },
            },
            {
                types: ["class-name", "builtin", "namespace", "type", "type-definition"],
                style: { color: c.type },
            },
            {
                types: ["attribute", "decorator", "lifetime-annotation", "annotation"],
                style: { color: c.type, fontStyle: "italic" },
            },
            {
                types: ["property", "attr-name", "variable", "parameter", "key", "table"],
                style: { color: c.ink },
            },
            {
                types: ["punctuation", "operator"],
                style: { color: c.punctuation },
            },
        ],
    };
}

export const boraLight = listing({
    ink: "#151515",
    paper: "#fdfdfb",
    comment: "#66655f",
    keyword: "#1d3f94",
    string: "#2a662a",
    number: "#8b2c55",
    func: "#5a3594",
    type: "#0b5f6a",
    punctuation: "#54534d",
});

export const boraDark = listing({
    ink: "#f1efe9",
    paper: "#151515",
    comment: "#9c9a92",
    keyword: "#93b2ff",
    string: "#91cf8e",
    number: "#eb9cc3",
    func: "#c6a9f6",
    type: "#7fcfd8",
    punctuation: "#b8b6ae",
});
