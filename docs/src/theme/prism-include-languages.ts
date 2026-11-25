/*
 * Swizzled from @docusaurus/theme-classic to add a BibTeX grammar, which
 * Prism does not ship. The rest loads `prism.additionalLanguages` exactly as
 * the original does.
 */

import siteConfig from "@generated/docusaurus.config";
import type * as PrismNamespace from "prismjs";

const bibtex = {
    comment: /%.*/,
    keyword: /@\w+/,
    "class-name": { pattern: /(@\w+\s*\{\s*)[^,\s{}]+/, lookbehind: true },
    property: /\b[\w-]+(?=\s*=)/,
    string: {
        pattern: /(=\s*)(?:\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|"[^"]*"|\d+)/,
        lookbehind: true,
        greedy: true,
    },
    punctuation: /[{},=]/,
};

export default function prismIncludeLanguages(PrismObject: typeof PrismNamespace): void {
    const {
        themeConfig: { prism },
    } = siteConfig;
    const { additionalLanguages } = prism as { additionalLanguages: string[] };

    // Prism components register on the global Prism; mount prism-react-renderer's
    // instance there while they load, then restore the global.
    const globals = globalThis as { Prism?: typeof PrismNamespace };
    const before = globals.Prism;
    globals.Prism = PrismObject;
    additionalLanguages.forEach((lang) => {
        require(`prismjs/components/prism-${lang}`);
    });
    PrismObject.languages.bibtex = bibtex;
    delete globals.Prism;
    if (before !== undefined) {
        globals.Prism = before;
    }
}
