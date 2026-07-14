import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const input =
  "/Users/aritra/Projects/smp_jetmass_run2/outputs/b2g_ttbarhadronic_progress_update_aritra_review_copy.pptx";
const output =
  "/Users/aritra/Projects/smp_jetmass_run2/outputs/b2g_ttbarhadronic_progress_update_aritra_polished_copy.pptx";
const previewDir =
  "/private/tmp/codex-presentations/b2g-review/polished-preview";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function shapeByName(slide, name) {
  const shape = slide.shapes.items.find((item) => item.name === name);
  if (!shape) throw new Error(`Shape not found: ${name}`);
  return shape;
}

function setShapeText(presentation, slideNumber, shapeName, text) {
  const slide = presentation.slides.getItem(slideNumber - 1);
  const shape = shapeByName(slide, shapeName);
  shape.text.set(text);
}

function replaceTextInSlide(presentation, slideNumber, replacements) {
  const slide = presentation.slides.getItem(slideNumber - 1);
  for (const shape of slide.shapes.items) {
    const current = String(shape.text ?? "");
    if (!current) continue;
    let updated = current;
    for (const [from, to] of replacements) {
      updated = updated.replaceAll(from, to);
    }
    if (updated !== current) shape.text = updated;
  }
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });

  const presentation = await PresentationFile.importPptx(
    await FileBlob.load(input),
  );

  setShapeText(
    presentation,
    1,
    "TextBox 2",
    "Resonant ttbar search\n(all-hadronic)",
  );
  setShapeText(
    presentation,
    1,
    "TextBox 3",
    "2024 JetMET, sqrt(s) = 13.6 TeV",
  );

  setShapeText(
    presentation,
    2,
    "TextBox 3",
    "Analysis strategy and samples\nEvent selection and categories\nGlobalParT3 top-tagging WPs\n2DAlphabet background estimate\nFit validation and systematics\nCurrent limits and next steps",
  );

  setShapeText(
    presentation,
    3,
    "TextBox 14",
    "Select boosted all-hadronic ttbar candidates with two AK8 jets\nDefine PASS and FAIL regions using GlobalParT3 top-tagging requirements\nSplit events into central (|Delta y| < 1.0) and forward (|Delta y| >= 1.0) categories\nEstimate multijet background with 2DAlphabet using R(mtt, mt)\nExtract signal with a binned maximum-likelihood fit in mtt and top-jet mass",
  );

  setShapeText(
    presentation,
    4,
    "TextBox 10",
    "Data\n2024 JetMET data (NanoAODv15), L = 109.95 fb^-1\n\nMonte Carlo\nSM ttbar: TTto4Q, POWHEG + PYTHIA8 at 13.6 TeV\nQCD multijet: pT-binned PYTHIA8, used for top-tagger WP derivation and validation\nSignal: Z' -> ttbar, MadGraph + PYTHIA8\n  Samples are binned in resonance mass and width.\n\nCurrently included signal model: Z' -> ttbar; RS KK gluon and DM benchmarks are not yet included",
  );

  setShapeText(
    presentation,
    5,
    "TextBox 8",
    "Trigger: HLT_PFHT1050, requiring PF-jet HT > 1050 GeV",
  );
  setShapeText(
    presentation,
    5,
    "TextBox 10",
    "Offline HT > 1400 GeV, chosen above the trigger turn-on\nAt least two selected AK8 jets; leading AK8 jet pT > 400 GeV\nBoosted back-to-back topology: |Delta phi(j1,j2)| > 2.1\nTop-tagging categories use GlobalParT3 at the 0.1% mistag WP",
  );

  setShapeText(
    presentation,
    6,
    "TextBox 10",
    "The events are categorized based on rapidity separation of the two selected AK8 jets\nThis split improves sensitivity by separating different S/B regimes\nThe regions are defined as:\nCentral: |Delta y| < 1.0\nForward: |Delta y| >= 1.0",
  );
  setShapeText(
    presentation,
    6,
    "TextBox 4",
    "Top-tagging regions\nPASS signal region: both jets pass the tight 0.1% GlobalParT3 WP\nFAIL control region: tag jet passes the tight WP; probe jet fails tight WP and lies in the sideband discriminator interval [0.5% WP, 0.1% WP)",
  );

  setShapeText(
    presentation,
    7,
    "TextBox 4",
    "GlobalParT3 is used for top tagging.\nOfficial Run-3 WPs are not yet available; analysis-specific WPs are derived.",
  );
  setShapeText(
    presentation,
    7,
    "TextBox 19",
    "Matched top jets and QCD jets show clear discriminator separation",
  );

  setShapeText(
    presentation,
    8,
    "TextBox 6",
    "WPs are derived using QCD simulation for target mistag rates\nThe 0.1% tight WP is used for the signal region\nMass stability is validated across discriminator intervals",
  );

  setShapeText(
    presentation,
    11,
    "TextBox 11",
    "QCD simulation is shown only as a shape diagnostic\nQCD MC is normalized to data minus SM ttbar for display only\nObserved differences motivate the data-driven multijet estimate",
  );

  setShapeText(
    presentation,
    12,
    "TextBox 9",
    "SM ttbar and QCD multijet are the dominant backgrounds\nSM ttbar is derived directly from simulation\nMultijet QCD is estimated with the data-driven 2DAlphabet PASS/FAIL method\nPASS: both jets pass the tight 0.1% top-tag WP\nFAIL: tag jet passes tight WP; probe jet lies in the discriminator sideband\nFor each central/forward category, derive R(mtt, mt) from sideband fits\nPredict Npass_QCD(mtt, mt) = R(mtt, mt) x Nfail_QCD(mtt, mt)",
  );

  setShapeText(
    presentation,
    13,
    "TextBox 9",
    "The transfer function R is parameterized as a function of mtt and mt\nCandidate functional forms are compared with F-tests",
  );
  setShapeText(
    presentation,
    13,
    "TextBox 3",
    "A simultaneous binned maximum-likelihood fit is performed with COMBINE to derive the TF parameters",
  );

  setShapeText(presentation, 14, "TextBox 2", "Systematic Uncertainties");
  setShapeText(
    presentation,
    14,
    "TextBox 7",
    "Systematic variations are incorporated as nuisance parameters\nKey parameters:\nJet energy scale and resolution\n  JEC tag: Summer24Prompt24_V2_MC\n  JER tag: verify official 2024 recommendation\nSM ttbar cross section\nTheory uncertainties: Q2 scales and PDFs\nPileup reweighting\nTransfer-function parameters and background model",
  );

  setShapeText(
    presentation,
    15,
    "TextBox 8",
    "F-tests compare nested transfer-function choices\n2x1 is preferred over 1x1 in both categories; 2x2 is not justified over 2x1\nNominal choice: 2x1 transfer function",
  );

  setShapeText(
    presentation,
    16,
    "TextBox 8",
    "Goodness-of-fit tests are performed for the selected 2x1 transfer function with the signal region masked",
  );
  setShapeText(presentation, 16, "TextBox 12", "Central 2x1: p = 0.485");
  setShapeText(presentation, 16, "TextBox 13", "Forward 2x1: p = 0.180");

  setShapeText(
    presentation,
    17,
    "TextBox 3",
    "Nominal 2x1 transfer-function parameters after the simultaneous PASS/FAIL fit\nParameter uncertainties are propagated as background-shape nuisances",
  );

  setShapeText(presentation, 21, "TextBox 2", "Expected limits");

  setShapeText(
    presentation,
    22,
    "TextBox 29",
    "Validate 10% and 30% Z' width samples\nFinalize nominal TF and alternate-function checks\nComplete pull/impact review and blinded fit validation\nProceed to unblinding after B2G/statistics review",
  );
  setShapeText(presentation, 22, "Rounded Rectangle 24", "TF + fits\nValidating other widths");
  setShapeText(presentation, 22, "Rounded Rectangle 26", "F-test\nDone");
  setShapeText(presentation, 22, "Rounded Rectangle 27", "Unblind and results\nPending review");

  for (const slideNumber of [9, 10, 24, 25]) {
    replaceTextInSlide(presentation, slideNumber, [
      ["Antitag", "FAIL"],
      ["antitag", "FAIL"],
      ["Two top-tagged", "PASS"],
      ["2-tag", "PASS"],
    ]);
  }

  const affectedSlides = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 24, 25,
  ];
  for (const slideNumber of affectedSlides) {
    const slide = presentation.slides.getItem(slideNumber - 1);
    await writeBlob(
      `${previewDir}/slide-${String(slideNumber).padStart(2, "0")}.png`,
      await presentation.export({ slide, format: "png", scale: 1 }),
    );
  }

  const montage = await presentation.export({
    format: "webp",
    montage: true,
    scale: 1,
  });
  await writeBlob(`${previewDir}/polished-montage.webp`, montage);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(output);

  console.log(output);
  console.log(previewDir);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
