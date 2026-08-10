import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");
const publication = JSON.parse(read("PUBLICATION.json"));
const packageJson = JSON.parse(read("package.json"));
const packageLock = JSON.parse(read("package-lock.json"));

const errors = [];
const requiredFiles = [
  "LICENSE.md",
  "LICENSE-CONTENT.md",
  "LICENSE-CODE",
  "NOTICE",
  "THIRD_PARTY_NOTICES.md",
  "CITATION.cff",
  "PUBLICATION.json",
];

for (const file of requiredFiles) {
  if (!fs.existsSync(path.join(root, file))) errors.push(`Missing ${file}`);
}

if (!/^\d{4}\.\d+$/.test(publication.edition)) {
  errors.push(`Invalid publication edition: ${publication.edition}`);
}
if (!/^\d{4}-\d{2}-\d{2}$/.test(publication.release_date)) {
  errors.push(`Invalid release date: ${publication.release_date}`);
}
if (!/^10\.5281\/zenodo\.\d+$/.test(publication.concept_doi)) {
  errors.push(`Invalid concept DOI: ${publication.concept_doi}`);
}
if (!/^10\.5281\/zenodo\.\d+$/.test(publication.version_doi)) {
  errors.push(`Invalid version DOI: ${publication.version_doi}`);
}
if (publication.content_license !== "CC-BY-NC-SA-4.0") {
  errors.push("Unexpected content license");
}
if (publication.code_license !== "Apache-2.0") {
  errors.push("Unexpected code license");
}

const citation = read("CITATION.cff");
if (!citation.includes(`version: "${publication.edition}"`)) {
  errors.push("CITATION.cff edition does not match PUBLICATION.json");
}
if (!citation.includes(`date-released: ${publication.release_date}`)) {
  errors.push("CITATION.cff release date does not match PUBLICATION.json");
}
if (!citation.includes(`doi: "${publication.version_doi}"`)) {
  errors.push("CITATION.cff DOI does not match PUBLICATION.json");
}

const notices = read("THIRD_PARTY_NOTICES.md");
const noticeNames = {
  katex: "KaTeX",
  "lucide-react": "Lucide React",
  react: "React",
  "react-dom": "React DOM",
  "react-markdown": "React Markdown",
  "rehype-katex": "rehype-katex",
  "remark-gfm": "remark-gfm",
  "remark-math": "remark-math",
};
for (const dependency of Object.keys(packageJson.dependencies)) {
  const resolved = packageLock.packages[`node_modules/${dependency}`];
  if (!resolved) {
    errors.push(`Dependency missing from package-lock.json: ${dependency}`);
    continue;
  }
  if (!notices.includes(`| ${noticeNames[dependency] ?? dependency} | ${resolved.version} |`)) {
    errors.push(`Third-party notice is missing ${dependency}@${resolved.version}`);
  }
}

const licenseMap = read("LICENSE.md");
for (const marker of ["CC BY-NC-SA 4.0", "Apache-2.0", "restricted Stepik course materials"]) {
  if (!licenseMap.includes(marker)) errors.push(`LICENSE.md is missing: ${marker}`);
}

if (errors.length) {
  console.error(JSON.stringify({ status: "FAIL", errors }, null, 2));
  process.exit(1);
}

console.log(
  JSON.stringify(
    {
      status: "PASS",
      edition: publication.edition,
      releaseDate: publication.release_date,
      conceptDoi: publication.concept_doi,
      versionDoi: publication.version_doi,
      requiredFiles: requiredFiles.length,
      declaredRuntimeDependencies: Object.keys(packageJson.dependencies).length,
    },
    null,
    2,
  ),
);
