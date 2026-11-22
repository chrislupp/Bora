import type { PrismTheme } from "prism-react-renderer";

/*
 * Code is set as a one-ink listing: keywords in bold, comments in italic, and
 * no colour. On this site colour belongs to the graph sheet, so structure in
 * code is carried by weight and slant.
 */
function listing(ink: string, paper: string): PrismTheme {
    return {
        plain: { color: ink, backgroundColor: paper },
        styles: [
            {
                types: ["comment", "prolog", "doctype", "cdata"],
                style: { fontStyle: "italic" },
            },
            {
                types: ["keyword", "atrule", "important", "tag", "section", "table"],
                style: { fontWeight: "bold" },
            },
            {
                types: ["macro", "attribute", "lifetime-annotation"],
                style: { fontStyle: "italic" },
            },
        ],
    };
}

export const boraLight = listing("#151515", "#fdfdfb");
export const boraDark = listing("#f1efe9", "#151515");
