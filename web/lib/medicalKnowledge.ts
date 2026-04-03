export type KnowledgeCategory =
  | "glycemic"
  | "lipids"
  | "liver"
  | "renal"
  | "thyroid"
  | "hematology"
  | "inflammation"
  | "vitamins"
  | "metabolic";

export type EvidenceLevel = "A" | "B" | "C" | "consensus";

export interface InterpretationBand {
  label: string;
  range: string;
  interpretation: string;
  typicalAction: string;
}

export interface KnowledgeEntry {
  id: string;
  category: KnowledgeCategory;
  testNames: string[];
  aliases: string[];
  keywords: string[];
  relatedTests: string[];
  units: string[];
  source: string;
  sourceUrl: string;
  title: string;
  content: string;
  interpretationBands: InterpretationBand[];
  trendSignals: string[];
  confounders: string[];
  escalationTriggers: string[];
  patientFriendlyActions: string[];
  evidenceLevel: EvidenceLevel;
  lastUpdated: string;
}

export const MEDICAL_KNOWLEDGE_BASE: KnowledgeEntry[] = [
  {
    id: "hba1c-ada-2024",
    category: "glycemic",
    testNames: ["hba1c", "glycated haemoglobin", "glycated hemoglobin"],
    aliases: ["a1c", "hemoglobin a1c", "glycohemoglobin"],
    keywords: ["diabetes", "prediabetes", "glycemic control", "average glucose"],
    relatedTests: ["fasting glucose", "postprandial glucose", "insulin", "urine albumin"],
    units: ["%"],
    source: "American Diabetes Association Standards of Care 2024",
    sourceUrl: "https://diabetesjournals.org/care/issue/47/Supplement_1",
    title: "HbA1C Targets and Diabetes Classification",
    content:
      "HbA1C reflects approximately 8-12 weeks of average glycemia. ADA classifies values below 5.7% as normoglycemia, 5.7-6.4% as prediabetes, and 6.5% or higher on two separate tests as diabetes. In established diabetes, most non-pregnant adults target below 7.0%, while stricter targets (for example below 6.5%) may be reasonable only when hypoglycemia risk is low and life expectancy is long. Persistently high HbA1C correlates with higher microvascular complication risk.",
    interpretationBands: [
      {
        label: "Normal",
        range: "<5.7%",
        interpretation: "No biochemical evidence of dysglycemia in the tested period.",
        typicalAction: "Continue routine prevention and periodic reassessment.",
      },
      {
        label: "Prediabetes",
        range: "5.7-6.4%",
        interpretation: "Elevated risk state with progression potential to diabetes.",
        typicalAction: "Lifestyle intervention and interval retesting are recommended.",
      },
      {
        label: "Diabetes range",
        range: ">=6.5%",
        interpretation: "Consistent with diabetes when confirmed by repeat testing.",
        typicalAction: "Needs clinician-led treatment planning and complication screening.",
      },
    ],
    trendSignals: [
      "A drop of >=0.5% over 3-6 months usually reflects meaningful glycemic improvement.",
      "A rising trajectory despite stable medication suggests adherence, dose, or lifestyle mismatch.",
      "Large oscillation between tests should prompt review of treatment consistency and acute illness periods.",
    ],
    confounders: [
      "Anemia, recent blood loss, hemoglobin variants, or kidney disease can bias HbA1C interpretation.",
      "Recent transfusion can invalidate short-term comparability.",
    ],
    escalationTriggers: [
      "HbA1C >=9% suggests poor control and high near-term risk of complications.",
      "HbA1C rise >=1% across sequential tests warrants urgent therapy review.",
    ],
    patientFriendlyActions: [
      "Track medication timing, meal pattern, and exercise consistency between tests.",
      "Discuss individualized target rather than using one universal number.",
    ],
    evidenceLevel: "A",
    lastUpdated: "2024-01",
  },
  {
    id: "fasting-glucose-ada-2024",
    category: "glycemic",
    testNames: ["fasting plasma glucose", "fasting glucose", "fbs"],
    aliases: ["fpg", "fasting blood sugar", "fasting sugar"],
    keywords: ["fasting", "glucose", "prediabetes", "diabetes diagnosis"],
    relatedTests: ["hba1c", "postprandial glucose", "insulin"],
    units: ["mg/dL", "mmol/L"],
    source: "American Diabetes Association Standards of Care 2024",
    sourceUrl: "https://diabetesjournals.org/care/issue/47/Supplement_1",
    title: "Fasting Plasma Glucose Diagnostic and Monitoring Ranges",
    content:
      "Fasting plasma glucose is a point-in-time marker and should be interpreted with fasting duration, illness context, and medication timing. ADA cutoffs are below 100 mg/dL for normal fasting glucose, 100-125 mg/dL for impaired fasting glucose (prediabetes), and 126 mg/dL or higher for diabetes when confirmed. Compared with HbA1C, fasting glucose can vary more day-to-day and is sensitive to sleep, stress, and acute infection.",
    interpretationBands: [
      {
        label: "Normal fasting",
        range: "<100 mg/dL",
        interpretation: "No fasting hyperglycemia detected at this time.",
        typicalAction: "Maintain preventive lifestyle and periodic monitoring.",
      },
      {
        label: "Impaired fasting glucose",
        range: "100-125 mg/dL",
        interpretation: "Prediabetes range indicating elevated progression risk.",
        typicalAction: "Structured lifestyle intervention and repeat testing.",
      },
      {
        label: "Diabetes threshold",
        range: ">=126 mg/dL",
        interpretation: "Diabetes range when confirmed on repeat measurement.",
        typicalAction: "Prompt clinician follow-up and treatment evaluation.",
      },
    ],
    trendSignals: [
      "Consistent rise over serial fasting tests is more informative than one isolated abnormal value.",
      "Early morning increases despite controlled HbA1C may indicate dawn phenomenon.",
    ],
    confounders: [
      "Short fasting window, steroid therapy, acute stress, and infection can elevate values.",
      "Very low carbohydrate intake can alter fasting interpretation when compared with prior baseline.",
    ],
    escalationTriggers: [
      "Repeated fasting values >=126 mg/dL after confirmation.",
      "Fasting glucose >180 mg/dL with symptoms of dehydration or polyuria.",
    ],
    patientFriendlyActions: [
      "Standardize fasting duration before repeat tests to improve comparability.",
      "Pair fasting trends with HbA1C to separate short-term spikes from long-term control.",
    ],
    evidenceLevel: "A",
    lastUpdated: "2024-01",
  },
  {
    id: "vitamin-d-endocrine-2024",
    category: "vitamins",
    testNames: ["25-oh vitamin d", "vitamin d", "25 hydroxy vitamin d"],
    aliases: ["25(oh)d", "calcidiol", "cholecalciferol status"],
    keywords: ["deficiency", "bone", "calcium", "muscle weakness", "supplementation"],
    relatedTests: ["calcium", "phosphorus", "alkaline phosphatase", "pth"],
    units: ["ng/mL", "nmol/L"],
    source: "Endocrine Society Clinical Practice Guideline 2024",
    sourceUrl: "https://www.endocrine.org/clinical-practice-guidelines",
    title: "Vitamin D Deficiency and Supplementation",
    content:
      "25-hydroxy vitamin D is used to assess vitamin D stores. Values below 20 ng/mL indicate deficiency and are linked to osteomalacia risk, muscle weakness, and fracture susceptibility. Values 20-29 ng/mL indicate insufficiency, while 30-100 ng/mL are typically considered sufficient for most adults. Retesting is usually done after approximately 3 months of replacement to evaluate response.",
    interpretationBands: [
      {
        label: "Deficiency",
        range: "<20 ng/mL",
        interpretation: "High likelihood of inadequate vitamin D stores with clinical impact risk.",
        typicalAction: "Therapeutic replacement and follow-up testing.",
      },
      {
        label: "Insufficiency",
        range: "20-29 ng/mL",
        interpretation: "Suboptimal level, often improved with maintenance supplementation.",
        typicalAction: "Dose optimization and lifestyle review (sun exposure, diet).",
      },
      {
        label: "Sufficient",
        range: "30-100 ng/mL",
        interpretation: "Generally adequate status for most adults.",
        typicalAction: "Continue maintenance if risk factors persist.",
      },
    ],
    trendSignals: [
      "Rise by 8-15 ng/mL over 8-12 weeks usually suggests adequate repletion response.",
      "Minimal change after supplementation should prompt adherence and malabsorption review.",
    ],
    confounders: [
      "Obesity, malabsorption syndromes, chronic kidney disease, and anticonvulsant therapy can blunt response.",
      "Seasonal variation and reduced sunlight exposure can lower baseline levels.",
    ],
    escalationTriggers: [
      "Level <10 ng/mL with musculoskeletal symptoms needs urgent clinician review.",
      "Persistently low values despite supplementation should trigger secondary-cause workup.",
    ],
    patientFriendlyActions: [
      "Take supplements consistently and repeat test after the interval advised by clinician.",
      "Discuss total daily dose and avoid unsupervised high-dose chronic use.",
    ],
    evidenceLevel: "B",
    lastUpdated: "2024-06",
  },
  {
    id: "ldl-aha-2023",
    category: "lipids",
    testNames: ["ldl cholesterol", "ldl", "low density lipoprotein"],
    aliases: ["ldl-c", "bad cholesterol"],
    keywords: ["cholesterol", "cardiovascular", "statin", "ascvd", "lipid"],
    relatedTests: ["hdl", "triglycerides", "non-hdl", "total cholesterol"],
    units: ["mg/dL"],
    source: "ACC/AHA Guideline on Management of Blood Cholesterol 2023",
    sourceUrl: "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000625",
    title: "LDL Cholesterol Targets and Risk Stratification",
    content:
      "LDL cholesterol is a major modifiable cardiovascular risk factor. Broad interpretation bands are below 100 mg/dL (optimal), 100-129 (near-optimal), 130-159 (borderline high), 160-189 (high), and 190 or higher (very high). Management intensity should align with absolute cardiovascular risk, diabetes status, and prior cardiovascular disease rather than LDL value alone.",
    interpretationBands: [
      {
        label: "Optimal",
        range: "<100 mg/dL",
        interpretation: "Lower atherogenic burden for most primary prevention settings.",
        typicalAction: "Maintain lifestyle and periodic monitoring.",
      },
      {
        label: "Borderline-high",
        range: "130-159 mg/dL",
        interpretation: "Elevated risk trajectory, especially with other risk factors.",
        typicalAction: "Structured lifestyle treatment and risk-based pharmacotherapy discussion.",
      },
      {
        label: "Very high",
        range: ">=190 mg/dL",
        interpretation: "Likely severe dyslipidemia; genetic causes should be considered.",
        typicalAction: "Prompt clinician evaluation and aggressive LDL-lowering strategy.",
      },
    ],
    trendSignals: [
      "Sustained LDL drop >=30% reflects moderate response; >=50% often expected with high-intensity therapy.",
      "Rising LDL after initial control can indicate adherence drift, diet change, or medication interruption.",
    ],
    confounders: [
      "Non-fasting sample and high triglycerides can affect calculated LDL reliability.",
      "Acute illness and major weight change can transiently alter lipid profile.",
    ],
    escalationTriggers: [
      "LDL >=190 mg/dL requires urgent risk and secondary-cause evaluation.",
      "Persistent LDL elevation despite therapy should trigger treatment intensification review.",
    ],
    patientFriendlyActions: [
      "Track saturated fat intake and medication adherence between tests.",
      "Discuss personalized LDL goal based on full cardiovascular risk profile.",
    ],
    evidenceLevel: "A",
    lastUpdated: "2023-11",
  },
  {
    id: "hdl-aha-2023",
    category: "lipids",
    testNames: ["hdl cholesterol", "hdl", "high density lipoprotein"],
    aliases: ["hdl-c", "good cholesterol"],
    keywords: ["protective", "lipid", "cardiovascular risk"],
    relatedTests: ["ldl", "triglycerides", "non-hdl"],
    units: ["mg/dL"],
    source: "ACC/AHA Guideline on Management of Blood Cholesterol 2023",
    sourceUrl: "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000625",
    title: "HDL Cholesterol in Risk Context",
    content:
      "Low HDL is associated with increased cardiovascular risk, but simply increasing HDL pharmacologically does not always reduce events. Typical risk bands are below 40 mg/dL for men and below 50 mg/dL for women as low, with above 60 mg/dL considered favorable. Interpretation should always be integrated with LDL, triglycerides, and global risk factors.",
    interpretationBands: [
      {
        label: "Low",
        range: "<40 mg/dL (men), <50 mg/dL (women)",
        interpretation: "Associated with increased long-term cardiovascular risk.",
        typicalAction: "Prioritize exercise, smoking cessation, and weight/metabolic optimization.",
      },
      {
        label: "Protective range",
        range: ">=60 mg/dL",
        interpretation: "Generally favorable marker when viewed with whole lipid profile.",
        typicalAction: "Maintain current preventive strategies.",
      },
    ],
    trendSignals: [
      "Gradual HDL increase can occur with sustained aerobic training and weight loss.",
      "Falling HDL with rising triglycerides may signal worsening insulin resistance.",
    ],
    confounders: [
      "Genetics strongly influences HDL; isolated low HDL may persist despite healthy lifestyle.",
      "Smoking and uncontrolled diabetes commonly suppress HDL.",
    ],
    escalationTriggers: [
      "Very low HDL with mixed dyslipidemia and diabetes features warrants comprehensive risk management.",
    ],
    patientFriendlyActions: [
      "Focus on risk reduction bundle, not HDL alone.",
      "Use trend with LDL and triglycerides to understand cardiometabolic direction.",
    ],
    evidenceLevel: "B",
    lastUpdated: "2023-11",
  },
  {
    id: "triglycerides-aha-2023",
    category: "lipids",
    testNames: ["triglycerides", "tg"],
    aliases: ["serum triglycerides"],
    keywords: ["hypertriglyceridemia", "pancreatitis", "metabolic syndrome", "lipid"],
    relatedTests: ["hdl", "ldl", "non-hdl", "fasting glucose"],
    units: ["mg/dL"],
    source: "ACC Expert Consensus Decision Pathway 2023",
    sourceUrl: "https://www.jacc.org",
    title: "Triglyceride Risk Tiers and Management",
    content:
      "Triglycerides are classified as normal (<150 mg/dL), borderline high (150-199), high (200-499), and very high (>=500). Very high levels increase pancreatitis risk and may require urgent dietary and pharmacologic intervention. Moderate elevations often travel with insulin resistance, obesity, and fatty liver, and should trigger broader metabolic risk management.",
    interpretationBands: [
      {
        label: "Normal",
        range: "<150 mg/dL",
        interpretation: "No significant hypertriglyceridemia.",
        typicalAction: "Maintain healthy diet and regular activity.",
      },
      {
        label: "High",
        range: "200-499 mg/dL",
        interpretation: "Associated with elevated cardiometabolic risk.",
        typicalAction: "Intensify lifestyle and evaluate secondary causes.",
      },
      {
        label: "Very high",
        range: ">=500 mg/dL",
        interpretation: "Substantial pancreatitis risk in susceptible patients.",
        typicalAction: "Urgent clinician follow-up and rapid reduction strategy.",
      },
    ],
    trendSignals: [
      "Rapid reduction after alcohol/sugar restriction suggests lifestyle-responsive component.",
      "Persistent severe elevation suggests secondary or genetic lipid disorder.",
    ],
    confounders: [
      "Recent meal, alcohol intake, uncontrolled diabetes, hypothyroidism, and certain drugs can elevate TG.",
    ],
    escalationTriggers: [
      "TG >=500 mg/dL or rise toward >=1000 mg/dL needs urgent clinical review.",
    ],
    patientFriendlyActions: [
      "Limit refined carbohydrate and alcohol intake before retesting.",
      "Ask about non-HDL and ApoB if residual risk remains high.",
    ],
    evidenceLevel: "A",
    lastUpdated: "2023-10",
  },
  {
    id: "sgot-ast-aasld-2023",
    category: "liver",
    testNames: ["aspartate aminotransferase", "sgot", "ast"],
    aliases: ["sgot/ast", "ast enzyme"],
    keywords: ["liver", "enzyme", "hepatitis", "fatty liver", "muscle injury"],
    relatedTests: ["alt", "alp", "bilirubin", "ggt"],
    units: ["U/L"],
    source: "AASLD Practice Guidance on Liver Enzymes 2023",
    sourceUrl: "https://www.aasld.org/publications/practice-guidelines",
    title: "AST Elevation Patterns",
    content:
      "AST should be interpreted with ALT and clinical context because AST can rise from hepatic and non-hepatic causes. Mild elevation (about 1-3 times upper limit) is commonly seen in fatty liver disease, alcohol exposure, medications, or strenuous exercise. Disproportionate AST elevation with AST:ALT ratio greater than 2 can suggest alcohol-related injury but is not diagnostic on its own.",
    interpretationBands: [
      {
        label: "Mild elevation",
        range: "1-3x upper reference limit",
        interpretation: "Often chronic, multifactorial, and non-urgent when isolated.",
        typicalAction: "Repeat panel and review metabolic, medication, and alcohol factors.",
      },
      {
        label: "Moderate elevation",
        range: "3-10x upper reference limit",
        interpretation: "Requires structured workup for active liver injury causes.",
        typicalAction: "Prompt clinician evaluation and expanded liver panel.",
      },
      {
        label: "Severe elevation",
        range: ">10x upper reference limit",
        interpretation: "May represent acute hepatocellular injury.",
        typicalAction: "Urgent assessment.",
      },
    ],
    trendSignals: [
      "Parallel AST and ALT decline usually indicates improving hepatocellular injury.",
      "Persistent isolated AST elevation should prompt muscle source consideration.",
    ],
    confounders: [
      "Heavy exercise, statin use, muscle disease, and hemolysis can elevate AST without major liver disease.",
    ],
    escalationTriggers: [
      "Severe enzyme elevation or associated jaundice/coagulopathy signs require urgent care.",
    ],
    patientFriendlyActions: [
      "Avoid alcohol and unnecessary hepatotoxic exposure before retest.",
      "Ask for full liver panel rather than interpreting AST in isolation.",
    ],
    evidenceLevel: "B",
    lastUpdated: "2023-08",
  },
  {
    id: "alt-aasld-2023",
    category: "liver",
    testNames: ["alanine aminotransferase", "sgpt", "alt"],
    aliases: ["sgpt/alt", "alt enzyme"],
    keywords: ["liver inflammation", "fatty liver", "hepatocellular"],
    relatedTests: ["ast", "alp", "bilirubin", "ggt"],
    units: ["U/L"],
    source: "AASLD Practice Guidance on Liver Enzymes 2023",
    sourceUrl: "https://www.aasld.org/publications/practice-guidelines",
    title: "ALT Interpretation and Follow-up",
    content:
      "ALT is generally more liver-specific than AST for hepatocellular injury patterns. Persistent mild ALT elevation is often associated with metabolic dysfunction-associated steatotic liver disease, alcohol use, and medication effects. Dynamic trend over serial tests is usually more clinically useful than one isolated value.",
    interpretationBands: [
      {
        label: "Borderline elevation",
        range: "Up to approximately 2x upper reference limit",
        interpretation: "Common and often chronic; needs metabolic context.",
        typicalAction: "Repeat testing and risk-factor review.",
      },
      {
        label: "Significant elevation",
        range: ">=3x upper reference limit",
        interpretation: "Higher probability of active hepatocellular injury.",
        typicalAction: "Prompt clinician-led etiologic evaluation.",
      },
    ],
    trendSignals: [
      "Sustained reduction after weight loss and glycemic improvement supports metabolic etiology.",
      "Rising ALT alongside triglycerides and glucose suggests worsening metabolic liver stress.",
    ],
    confounders: [
      "Medication changes, herbal products, and recent viral illness can transiently elevate ALT.",
    ],
    escalationTriggers: [
      "ALT >10x reference limit or abrupt steep rise requires urgent clinical assessment.",
    ],
    patientFriendlyActions: [
      "Track weight, alcohol, and medication changes between liver panels.",
      "Discuss whether imaging or fibrosis risk assessment is appropriate.",
    ],
    evidenceLevel: "B",
    lastUpdated: "2023-08",
  },
  {
    id: "creatinine-kdigo-2024",
    category: "renal",
    testNames: ["serum creatinine", "creatinine"],
    aliases: ["scr", "blood creatinine"],
    keywords: ["kidney function", "renal", "ckd", "egfr"],
    relatedTests: ["egfr", "urea", "urine albumin", "potassium"],
    units: ["mg/dL"],
    source: "KDIGO CKD Guideline Update 2024",
    sourceUrl: "https://kdigo.org/guidelines/",
    title: "Creatinine as a Kidney Function Marker",
    content:
      "Creatinine is used with age, sex, and race-independent equations to estimate GFR. Isolated creatinine values should not be interpreted without trend and eGFR context. Small absolute creatinine changes can represent meaningful kidney function decline, especially in patients with low baseline muscle mass or chronic comorbidities.",
    interpretationBands: [
      {
        label: "Within lab reference",
        range: "Lab-specific",
        interpretation: "May still coexist with reduced eGFR in older adults.",
        typicalAction: "Always pair with eGFR and urine albumin metrics.",
      },
      {
        label: "Above reference",
        range: "Lab-specific",
        interpretation: "Possible reduced filtration or reversible acute causes.",
        typicalAction: "Review trend, hydration status, and nephrotoxic exposures.",
      },
    ],
    trendSignals: [
      "Persistent upward trend over months is more concerning than one isolated borderline result.",
      "Acute rise after dehydration or illness may partially reverse with treatment.",
    ],
    confounders: [
      "Muscle mass, dehydration, high meat intake, and some drugs can alter creatinine independent of intrinsic CKD progression.",
    ],
    escalationTriggers: [
      "Rapid creatinine rise (for example >=0.3 mg/dL in short interval) needs urgent review.",
      "Worsening creatinine with oliguria, edema, or hyperkalemia symptoms requires urgent care.",
    ],
    patientFriendlyActions: [
      "Track blood pressure, diabetes control, and medication safety with clinician.",
      "Avoid unsupervised NSAID overuse in kidney-risk settings.",
    ],
    evidenceLevel: "A",
    lastUpdated: "2024-02",
  },
  {
    id: "egfr-kdigo-2024",
    category: "renal",
    testNames: ["egfr", "estimated gfr", "estimated glomerular filtration rate"],
    aliases: ["gfr", "ckd-epi egfr"],
    keywords: ["ckd stage", "kidney stage", "renal decline", "albuminuria"],
    relatedTests: ["creatinine", "urine acr", "urea", "potassium"],
    units: ["mL/min/1.73m2"],
    source: "KDIGO CKD Guideline Update 2024",
    sourceUrl: "https://kdigo.org/guidelines/",
    title: "eGFR Staging and Risk Context",
    content:
      "eGFR should be interpreted as a trend and in combination with urine albumin category. Typical CKD staging uses G1 >=90, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, and G5 <15 mL/min/1.73m2. Persistent reduction for at least 3 months is required for chronic kidney disease diagnosis.",
    interpretationBands: [
      {
        label: "Mildly reduced",
        range: "60-89",
        interpretation: "May be age-related or early CKD depending on albuminuria and persistence.",
        typicalAction: "Monitor trend and assess urine albumin.",
      },
      {
        label: "Moderate reduction",
        range: "30-59",
        interpretation: "Clinically meaningful filtration decline with higher progression risk.",
        typicalAction: "Risk-factor control and structured CKD follow-up.",
      },
      {
        label: "Severe reduction",
        range: "<30",
        interpretation: "High-risk CKD stage requiring close specialist coordination.",
        typicalAction: "Prompt nephrology-aligned care planning.",
      },
    ],
    trendSignals: [
      "Annual decline slope is important for progression risk and treatment urgency.",
      "Short-term fluctuations can reflect hydration or intercurrent illness.",
    ],
    confounders: [
      "Acute kidney injury periods should not be confused with stable CKD trend.",
      "Body composition extremes can affect creatinine-derived estimates.",
    ],
    escalationTriggers: [
      "eGFR decline >5 mL/min/1.73m2 per year is concerning for progression.",
      "eGFR <30 with electrolyte abnormalities needs urgent clinical review.",
    ],
    patientFriendlyActions: [
      "Track blood pressure and diabetes targets to slow progression.",
      "Ask whether urine albumin testing is being done alongside eGFR.",
    ],
    evidenceLevel: "A",
    lastUpdated: "2024-02",
  },
  {
    id: "tsh-ata-2023",
    category: "thyroid",
    testNames: ["thyroid stimulating hormone", "tsh"],
    aliases: ["thyrotropin"],
    keywords: ["hypothyroid", "hyperthyroid", "subclinical", "thyroid"],
    relatedTests: ["free t4", "free t3", "anti-tpo"],
    units: ["mIU/L", "uIU/mL"],
    source: "American Thyroid Association Guidelines 2023",
    sourceUrl: "https://www.thyroid.org/professionals/ata-professional-guidelines",
    title: "TSH Reference Framework",
    content:
      "TSH is the primary screening marker for most thyroid dysfunction contexts. Typical reference interval is roughly 0.5-4.5 mIU/L, with lab variation. Elevated TSH with normal free T4 suggests subclinical hypothyroidism, while suppressed TSH with normal free T4 may indicate subclinical hyperthyroidism. Clinical decision-making depends on symptoms, age, antibody status, and cardiovascular risk.",
    interpretationBands: [
      {
        label: "Reference range",
        range: "Approximately 0.5-4.5 mIU/L",
        interpretation: "Thyroid axis likely euthyroid in most contexts.",
        typicalAction: "Routine follow-up if clinically indicated.",
      },
      {
        label: "Mild elevation",
        range: ">4.5 to <10 mIU/L",
        interpretation: "Possible subclinical hypothyroidism depending on free T4.",
        typicalAction: "Repeat test and assess symptoms/antibodies.",
      },
      {
        label: "Marked elevation",
        range: ">=10 mIU/L",
        interpretation: "Higher likelihood of clinically significant hypothyroidism.",
        typicalAction: "Prompt clinician review for treatment planning.",
      },
    ],
    trendSignals: [
      "Persistent TSH drift in one direction across multiple tests has higher clinical value than single borderline result.",
      "Normalization after therapy usually occurs gradually over weeks.",
    ],
    confounders: [
      "Biotin supplements, non-thyroidal illness, and assay interference can distort values.",
      "Pregnancy uses different reference targets.",
    ],
    escalationTriggers: [
      "TSH suppression with arrhythmia symptoms or marked elevation with severe symptoms needs urgent review.",
    ],
    patientFriendlyActions: [
      "Keep testing conditions and medication timing consistent before repeat thyroid panels.",
      "Interpret TSH together with free T4, not in isolation when values are borderline.",
    ],
    evidenceLevel: "A",
    lastUpdated: "2023-05",
  },
  {
    id: "free-t4-ata-2023",
    category: "thyroid",
    testNames: ["free t4", "free thyroxine", "ft4"],
    aliases: ["thyroxine free"],
    keywords: ["thyroxine", "overt hypothyroidism", "overt hyperthyroidism"],
    relatedTests: ["tsh", "free t3"],
    units: ["ng/dL", "pmol/L"],
    source: "American Thyroid Association Guidelines 2023",
    sourceUrl: "https://www.thyroid.org/professionals/ata-professional-guidelines",
    title: "Free T4 Confirmation Layer for Thyroid Status",
    content:
      "Free T4 helps confirm whether TSH abnormalities are subclinical or overt thyroid dysfunction. Low free T4 with high TSH supports overt hypothyroidism, while high free T4 with suppressed TSH supports overt hyperthyroidism. Borderline TSH values with normal free T4 often represent subclinical disease states and need trend-based follow-up.",
    interpretationBands: [
      {
        label: "Low FT4",
        range: "Below lab lower reference",
        interpretation: "Suggests overt hypothyroid physiology when TSH is elevated.",
        typicalAction: "Clinician-led management and dose planning.",
      },
      {
        label: "High FT4",
        range: "Above lab upper reference",
        interpretation: "Suggests overt hyperthyroid physiology when TSH is suppressed.",
        typicalAction: "Prompt endocrine review.",
      },
    ],
    trendSignals: [
      "FT4 normalization lag versus TSH can occur during treatment titration.",
    ],
    confounders: [
      "Biotin, assay interference, and major illness can affect free hormone measurements.",
    ],
    escalationTriggers: [
      "Marked FT4 elevation with tachycardia, tremor, or weight loss symptoms requires urgent review.",
    ],
    patientFriendlyActions: [
      "Use TSH and FT4 together to understand whether abnormality is mild or overt.",
    ],
    evidenceLevel: "B",
    lastUpdated: "2023-05",
  },
  {
    id: "hemoglobin-who-2024",
    category: "hematology",
    testNames: ["hemoglobin", "haemoglobin", "hb"],
    aliases: ["hgb"],
    keywords: ["anemia", "fatigue", "blood count", "oxygen carrying"],
    relatedTests: ["ferritin", "mcv", "rdw", "vitamin b12"],
    units: ["g/dL"],
    source: "WHO Haemoglobin Concentrations for Anaemia Diagnosis 2024 update",
    sourceUrl: "https://www.who.int/publications",
    title: "Hemoglobin Thresholds and Anemia Severity",
    content:
      "Hemoglobin interpretation depends on sex, age, pregnancy status, and altitude adjustments where relevant. In many adult contexts, values below approximately 13 g/dL in men and below 12 g/dL in women are considered anemic. Severity grading and trend should guide urgency, especially if symptoms or comorbid cardiac disease are present.",
    interpretationBands: [
      {
        label: "Mild anemia",
        range: "Near threshold below reference",
        interpretation: "Possible early or compensated anemia state.",
        typicalAction: "Etiology workup (iron, B12, folate, chronic disease markers).",
      },
      {
        label: "Moderate/severe anemia",
        range: "Substantially below threshold",
        interpretation: "Higher risk of symptoms and organ strain.",
        typicalAction: "Prompt diagnostic and treatment planning.",
      },
    ],
    trendSignals: [
      "Falling hemoglobin over serial tests is more significant than one low-normal reading.",
      "Rise after replacement therapy supports deficiency-mediated etiology.",
    ],
    confounders: [
      "Hydration status, recent bleeding, and chronic inflammation can influence readings.",
    ],
    escalationTriggers: [
      "Rapid hemoglobin drop or severe low values with dyspnea/chest symptoms requires urgent care.",
    ],
    patientFriendlyActions: [
      "Do not self-treat with iron unless deficiency is confirmed.",
      "Track symptoms alongside numeric trend.",
    ],
    evidenceLevel: "A",
    lastUpdated: "2024-03",
  },
  {
    id: "platelet-ash-2023",
    category: "hematology",
    testNames: ["platelet count", "platelets", "thrombocyte count"],
    aliases: ["plt", "thrombocytes"],
    keywords: ["bleeding risk", "thrombocytopenia", "thrombocytosis", "clot risk"],
    relatedTests: ["wbc", "hemoglobin", "peripheral smear"],
    units: ["x10^3/uL", "10^9/L"],
    source: "American Society of Hematology Guidance 2023",
    sourceUrl: "https://www.hematology.org/education/clinicians/guidelines-quality",
    title: "Platelet Count Clinical Context",
    content:
      "Platelet counts are broadly interpreted as low below around 150 x10^3/uL, normal in mid reference range, and high above around 450 x10^3/uL. Mild isolated abnormalities can be reactive (infection, inflammation, iron deficiency) and may normalize. Persistent or severe deviations need targeted hematology workup.",
    interpretationBands: [
      {
        label: "Thrombocytopenia",
        range: "<150 x10^3/uL",
        interpretation: "Potential bleeding-risk state depending on degree and cause.",
        typicalAction: "Repeat count and etiology assessment.",
      },
      {
        label: "Thrombocytosis",
        range: ">450 x10^3/uL",
        interpretation: "May be reactive or clonal; persistence matters.",
        typicalAction: "Evaluate inflammatory/reactive causes and trend.",
      },
    ],
    trendSignals: [
      "Persistent trajectory (not one isolated test) drives clinical significance.",
      "Falling platelets with infection or drug exposure may indicate acute process.",
    ],
    confounders: [
      "Lab artifact (platelet clumping) can cause false low values; smear confirmation may be needed.",
    ],
    escalationTriggers: [
      "Platelets <50 x10^3/uL or active bleeding symptoms need urgent evaluation.",
      "Very high persistent platelets with thrombotic symptoms needs urgent review.",
    ],
    patientFriendlyActions: [
      "Repeat abnormal counts before concluding chronic disorder unless clinically urgent.",
    ],
    evidenceLevel: "B",
    lastUpdated: "2023-07",
  },
  {
    id: "ferritin-bsh-2023",
    category: "hematology",
    testNames: ["ferritin", "serum ferritin"],
    aliases: ["iron store ferritin"],
    keywords: ["iron deficiency", "anemia workup", "inflammation marker"],
    relatedTests: ["hemoglobin", "transferrin saturation", "crp", "mcv"],
    units: ["ng/mL", "ug/L"],
    source: "British Society for Haematology Iron Deficiency Guidance 2023",
    sourceUrl: "https://b-s-h.org.uk/guidelines",
    title: "Ferritin in Iron Deficiency Workup",
    content:
      "Ferritin is the primary marker of iron stores, but it is also an acute phase reactant. Low ferritin is highly specific for iron deficiency, while normal or mildly elevated ferritin may not exclude deficiency when inflammation is present. Interpretation should be integrated with CRP, transferrin saturation, and red cell indices.",
    interpretationBands: [
      {
        label: "Low ferritin",
        range: "Below lab lower reference (often <30 ng/mL in many contexts)",
        interpretation: "Strong evidence of depleted iron stores.",
        typicalAction: "Iron deficiency evaluation and treatment planning.",
      },
      {
        label: "Normal-high ferritin with inflammation",
        range: "Within or above reference",
        interpretation: "May mask iron deficiency if inflammatory burden is high.",
        typicalAction: "Use additional iron markers and inflammation context.",
      },
    ],
    trendSignals: [
      "Ferritin rise with improving hemoglobin suggests replenishment success.",
      "Persistently low ferritin despite therapy suggests intake/adherence/absorption issues.",
    ],
    confounders: [
      "Inflammation, liver disease, and metabolic syndrome can elevate ferritin independent of iron stores.",
    ],
    escalationTriggers: [
      "Severe iron deficiency anemia pattern with symptoms needs prompt treatment.",
    ],
    patientFriendlyActions: [
      "Take iron only under clinician guidance and recheck objective response.",
    ],
    evidenceLevel: "B",
    lastUpdated: "2023-09",
  },
  {
    id: "vitamin-b12-efns-2024",
    category: "vitamins",
    testNames: ["vitamin b12", "cobalamin", "b12"],
    aliases: ["serum b12"],
    keywords: ["neuropathy", "macrocytosis", "fatigue", "deficiency"],
    relatedTests: ["hemoglobin", "mcv", "folate", "homocysteine"],
    units: ["pg/mL"],
    source: "EFNS Clinical Guidance on B12 Deficiency 2024",
    sourceUrl: "https://www.ean.org/guideline-reference-center",
    title: "Vitamin B12 Deficiency Interpretation",
    content:
      "Vitamin B12 deficiency can present with hematologic and neurologic features, and neurologic symptoms may precede severe anemia. Borderline serum levels may require confirmatory markers (for example methylmalonic acid or homocysteine) depending on context. Trend and symptom correlation are essential for clinically meaningful interpretation.",
    interpretationBands: [
      {
        label: "Likely deficiency",
        range: "Commonly <200 pg/mL (lab/context dependent)",
        interpretation: "High likelihood of deficiency state.",
        typicalAction: "Replacement and etiology assessment.",
      },
      {
        label: "Borderline range",
        range: "Approximately 200-300 pg/mL",
        interpretation: "Indeterminate in some contexts.",
        typicalAction: "Use confirmatory tests and clinical correlation.",
      },
    ],
    trendSignals: [
      "Rising B12 after replacement with symptom stabilization suggests response.",
      "No response should prompt adherence, absorption, and diagnosis reassessment.",
    ],
    confounders: [
      "Recent supplementation can normalize serum level without immediate tissue recovery.",
    ],
    escalationTriggers: [
      "Deficiency with neurologic symptoms requires timely treatment to limit irreversible deficits.",
    ],
    patientFriendlyActions: [
      "Report numbness, tingling, or gait change early during low B12 workup.",
    ],
    evidenceLevel: "B",
    lastUpdated: "2024-02",
  },
  {
    id: "hs-crp-aha-2023",
    category: "inflammation",
    testNames: ["hs-crp", "high sensitivity c-reactive protein", "crp"],
    aliases: ["high sensitivity crp", "c-reactive protein"],
    keywords: ["inflammation", "cardiovascular risk", "acute infection"],
    relatedTests: ["ldl", "triglycerides", "ferritin"],
    units: ["mg/L"],
    source: "AHA/CDC Scientific Statement on CRP 2023",
    sourceUrl: "https://www.ahajournals.org",
    title: "hs-CRP Risk Interpretation",
    content:
      "For cardiovascular risk framing, hs-CRP is often interpreted as low risk below 1 mg/L, average risk 1-3 mg/L, and higher risk above 3 mg/L. Values above 10 mg/L usually indicate acute inflammatory or infectious processes and are not used for baseline cardiovascular risk stratification until rechecked after recovery.",
    interpretationBands: [
      {
        label: "Lower inflammatory risk",
        range: "<1 mg/L",
        interpretation: "Lower baseline inflammatory signal.",
        typicalAction: "Continue risk-factor prevention strategy.",
      },
      {
        label: "Higher inflammatory risk",
        range: ">3 mg/L",
        interpretation: "Elevated inflammatory burden and increased cardiometabolic risk context.",
        typicalAction: "Assess reversible contributors and broader risk profile.",
      },
    ],
    trendSignals: [
      "Persistent elevation across repeated stable-health measurements is more meaningful than one spike.",
      "Drop after lifestyle optimization can parallel cardiometabolic risk improvement.",
    ],
    confounders: [
      "Acute infection, injury, and autoimmune flare can transiently elevate hs-CRP.",
    ],
    escalationTriggers: [
      "hs-CRP >10 mg/L should trigger repeat testing after acute illness resolves.",
    ],
    patientFriendlyActions: [
      "Repeat hs-CRP when free of acute infection to interpret chronic risk accurately.",
    ],
    evidenceLevel: "A",
    lastUpdated: "2023-03",
  },
  {
    id: "uric-acid-acr-2023",
    category: "metabolic",
    testNames: ["uric acid", "serum urate"],
    aliases: ["urate"],
    keywords: ["gout", "hyperuricemia", "metabolic risk", "renal stone"],
    relatedTests: ["creatinine", "egfr", "crp"],
    units: ["mg/dL"],
    source: "American College of Rheumatology Gout Guideline 2023",
    sourceUrl: "https://rheumatology.org",
    title: "Serum Uric Acid and Gout Risk Context",
    content:
      "Hyperuricemia increases gout and urate crystal deposition risk, but diagnosis of gout is clinical and not made by uric acid level alone. Persistent elevation with joint symptoms or nephrolithiasis history has higher significance. Target urate thresholds are typically lower in treated gout to prevent recurrent flares.",
    interpretationBands: [
      {
        label: "Within typical range",
        range: "Lab-specific reference",
        interpretation: "Lower urate burden at this time.",
        typicalAction: "Monitor if clinical risk factors exist.",
      },
      {
        label: "Hyperuricemia",
        range: "Above lab upper reference",
        interpretation: "Higher crystal and gout risk over time.",
        typicalAction: "Evaluate symptoms, kidney status, and lifestyle contributors.",
      },
    ],
    trendSignals: [
      "Persistent decline with urate-lowering therapy indicates better flare prevention alignment.",
      "Rising urate with CKD progression often needs medication and diet review.",
    ],
    confounders: [
      "Diuretics, alcohol, dehydration, and purine-heavy diet can elevate urate.",
    ],
    escalationTriggers: [
      "High urate with recurrent inflammatory arthritis symptoms needs timely rheumatology assessment.",
    ],
    patientFriendlyActions: [
      "Hydration and trigger-food moderation can support urate control alongside medical care.",
    ],
    evidenceLevel: "B",
    lastUpdated: "2023-11",
  },
];
