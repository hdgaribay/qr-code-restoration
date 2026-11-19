#!/usr/bin/env python3
"""Build an editable Stage 1/2 progress report from measured project outputs."""
from pathlib import Path
import csv
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'deliverables'/'ArtifactB_Stage1_Stage2_Progress_Report.docx'
LOGS=ROOT/'logs'; FIGS=ROOT/'figures'
OUT.parent.mkdir(parents=True,exist_ok=True)

def rows(filename):
    with (LOGS/filename).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def pct(x,d=1):return f'{100*float(x):.{d}f}%'
def table_borders(table):
    tblPr=table._tbl.tblPr
    borders=tblPr.first_child_found_in('w:tblBorders')
    if borders is None:
        borders=OxmlElement('w:tblBorders');tblPr.append(borders)
    for edge in ('top','left','bottom','right','insideH','insideV'):
        el=borders.find(qn('w:'+edge))
        if el is None:el=OxmlElement('w:'+edge);borders.append(el)
        el.set(qn('w:val'),'single');el.set(qn('w:sz'),'5');el.set(qn('w:space'),'0');el.set(qn('w:color'),'D9D9D9')
    for row in table.rows:
        for cell in row.cells:
            tcMar=OxmlElement('w:tcMar')
            for side,val in [('top','80'),('start','90'),('bottom','80'),('end','90')]:
                e=OxmlElement('w:'+side);e.set(qn('w:w'),val);e.set(qn('w:type'),'dxa');tcMar.append(e)
            cell._tc.get_or_add_tcPr().append(tcMar)
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
def set_repeat_table_header(row):
    trPr=row._tr.get_or_add_trPr();el=OxmlElement('w:tblHeader');el.set(qn('w:val'),'true');trPr.append(el)
def add_table(doc,headers,body,widths=None,font=8.3):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    if widths:
        for c,w in zip(t.columns,widths):c.width=Inches(w)
    for i,h in enumerate(headers):t.rows[0].cells[i].text=str(h)
    set_repeat_table_header(t.rows[0])
    for rec in body:
        cells=t.add_row().cells
        for i,val in enumerate(rec):cells[i].text=str(val)
    table_borders(t)
    for ri,row in enumerate(t.rows):
        for ci,cell in enumerate(row.cells):
            if ri==0:
                shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'17365D');cell._tc.get_or_add_tcPr().append(shd)
            elif ri%2==0:
                shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'F3F6FA');cell._tc.get_or_add_tcPr().append(shd)
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0)
                if ci>0:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.name='Aptos';run.font.size=Pt(font);run.font.color.rgb=RGBColor(255,255,255) if ri==0 else RGBColor(20,30,42)
                    if ri==0:run.bold=True
    return t
def add_caption(doc,text):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(10)
    r=p.add_run(text);r.italic=True;r.font.size=Pt(9);r.font.color.rgb=RGBColor(75,85,95)
def add_figure(doc,path,width=6.7,caption=''):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(1)
    p.add_run().add_picture(str(path),width=Inches(width))
    if caption:add_caption(doc,caption)
def bullet(doc,text):
    p=doc.add_paragraph(style='List Bullet');p.paragraph_format.space_after=Pt(3);p.add_run(text)
def heading(doc,text,level=1):
    p=doc.add_heading(text,level=level);p.paragraph_format.keep_with_next=True;return p

stage1=rows('stage1_synthetic_validation_comparison.csv')
stage2=rows('method_comparison_validation.csv')
hyp=rows('hypothesis_contrasts_validation.csv')
def s1(method,cond='ALL'):return next(r for r in stage1 if r['method']==method and r['impairments']==cond)
def s2(method,sev,cond='ALL'):return next(r for r in stage2 if r['method']==method and int(r['severity_level'])==sev and r['condition']==cond)
def hrow(h,sev,cond='ALL'):return next(r for r in hyp if r['hypothesis']==h and int(r['severity_level'])==sev and r['condition']==cond)

doc=Document();sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11);sec.left_margin=Inches(.78);sec.right_margin=Inches(.78);sec.top_margin=Inches(.7);sec.bottom_margin=Inches(.7)
styles=doc.styles
for sty in ('Normal','Title','Subtitle','Heading 1','Heading 2'):
    styles[sty].font.name='Aptos'
    styles[sty]._element.rPr.rFonts.set(qn('w:ascii'),'Aptos');styles[sty]._element.rPr.rFonts.set(qn('w:hAnsi'),'Aptos')
    styles[sty].font.color.rgb=RGBColor(0,0,0)
styles['Normal'].font.size=Pt(10.2);styles['Normal'].paragraph_format.space_after=Pt(6);styles['Normal'].paragraph_format.line_spacing=1.08
styles['Title'].font.size=Pt(23);styles['Title'].font.bold=True;styles['Title'].paragraph_format.space_after=Pt(4)
styles['Subtitle'].font.size=Pt(11);styles['Subtitle'].font.color.rgb=RGBColor(65,75,85);styles['Subtitle'].paragraph_format.space_after=Pt(5)
styles['Heading 1'].font.size=Pt(15);styles['Heading 1'].font.bold=True;styles['Heading 1'].paragraph_format.space_before=Pt(12);styles['Heading 1'].paragraph_format.space_after=Pt(5)
styles['Heading 2'].font.size=Pt(11.5);styles['Heading 2'].font.bold=True;styles['Heading 2'].paragraph_format.space_before=Pt(8);styles['Heading 2'].paragraph_format.space_after=Pt(3)

p=doc.add_paragraph(style='Title');p.add_run('ECE 253 Artifact B Stage 1 and Stage 2 Progress Report')
p=doc.add_paragraph(style='Subtitle');p.add_run('Synthetic pretraining, fine-tuning, and validation-only U-Net comparisons')
p=doc.add_paragraph();p.paragraph_format.space_after=Pt(10);r=p.add_run('November 13, 2026  |  UCSD ECE 253');r.bold=True;r.font.size=Pt(9)

doc.add_paragraph('The Stage 1 and Stage 2 training runs are complete, but the measured results do not support a broad improvement claim for the fine-tuned U-Nets. Combined-impairment training improved U-Net exact decoding at moderate shadow severity, while the gains disappeared at stronger severity. Structure-weighted loss did not improve exact decode rate. On the controlled common-ROI validation, classical restoration substantially outperformed the fine-tuned U-Nets at L3 and L4. The held-out test set was not accessed.')

heading(doc,'Study scope and safeguards')
doc.add_paragraph('This report adds the current Stage 1/2 results to the existing Artifact B research record. The earlier project and its report, checkpoints, manifests, and results remain in the package unchanged. This phase uses new synthetic Stage 1 identities and the existing locked Stage 2 training and validation partitions only.')
add_table(doc,['Split or source','Pairs / images','Use in this phase'],[
    ('Stage 1 generated training','6,000 SINGLE + 6,000 ALL','Pretraining only; 2,000 unique identities per arm'),
    ('Stage 1 generated validation','1,400','Checkpoint choice and evaluation; 200 disjoint identities'),
    ('Stage 2 existing training pool','693 usable of 940','Fine-tuning; 36 of 45 locked training identities'),
    ('Stage 2 existing validation','213 geometry-valid of 220','Selection and evaluation; all 220 retained in decode denominator'),
    ('Held-out test identities','0 image bytes read','Not used for training, selection, or evaluation')
],[1.65,1.55,3.75],font=8.5)
doc.add_paragraph('All Stage 2 methods use the same input-only detector-derived square ROI, the same ZXing-C++ exact-payload decoder, and no oracle geometry or ground-truth payload. Ten locked validation identities are used for identity-cluster bootstrap intervals. Seven Stage 2 validation images did not pass geometry checks; they remain decode failures in the full denominator.')

heading(doc,'Training design')
add_table(doc,['Model branch','Stage 1 data','Stage 2 initialization and data','Training budget'],[
    ('U-NET SINGLE','S, D, G only; 6,000 pairs','Corresponding Stage 1 checkpoint; same 693 Stage 2 pairs','240 updates per stage'),
    ('U-NET ALL','S, D, G, SD, SG, DG, SDG; 6,000 pairs','Corresponding Stage 1 checkpoint; same 693 Stage 2 pairs','240 updates per stage'),
    ('U-NET ALL + QR loss','Same ALL pairs and matched initialization','Stage 1 ALL-standard initialization; same 693 Stage 2 pairs','240 updates per stage')
],[1.35,2.25,2.55,.8],font=8.2)
doc.add_paragraph('Every model uses the Proposal four-scale Compact U-Net (16/32/64/128 channels, grayscale 256 × 256 input), AdamW, batch size 8, validation every 80 updates, and a fixed 240-update budget per stage. Matched branches share initialization seeds, sample order, and geometric augmentation randomization. SINGLE and ALL use separate randomly generated training identity pools, so H1 also contains between-pool identity variation; H2 uses matched data. Stage 1 payload inventories have zero overlap across both training arms and validation, and zero overlap with the existing validation/test payload inventory.')
doc.add_paragraph('The source pool contained 940 training synthetic images. Only 693 pairs from 36 identities passed the paired Clean-reference dual-decoder audit and input-only geometry checks; 247 images were excluded. The remaining nine training identities could not be used under the stated target/geometry requirements. The resulting Stage 2 result is therefore a fine-tune on an audited subset, not on all 45 identities.')

heading(doc,'Stage 1 synthetic pretraining results')
doc.add_paragraph('The Stage 1 validation images use independently generated QR matrices, so MER and region-error measurements are valid for this split. RAW already decodes 99.14% of the validation set, leaving little decode headroom. Both ALL models reached 99.93% DR; the structure-weighted model had slightly smaller MER, but the absolute errors were already near zero and DR did not change.')
s1_table=[]
for m,label in [('RAW','RAW'),('UNET_SINGLE_STANDARD','U-NET SINGLE'),('UNET_ALL_STANDARD','U-NET ALL'),('UNET_ALL_STRUCTURE_WEIGHTED','U-NET ALL + QR loss')]:
    r=s1(m);s1_table.append((label,f"{r['exact_decodes']}/1,400 ({pct(r['exact_decode_rate'],2)})",pct(r['module_error_rate'],4),
        pct(r['fixed_pattern_error_rate'],3),f"{pct(r['finder_error_rate'],3)} / {pct(r['timing_error_rate'],3)} / {pct(r['alignment_error_rate'],3)}",f"{float(r['mean_latency_ms']):.1f}"))
add_table(doc,['Method','Exact DR','MER','Fixed pattern errors','Finder / timing / alignment errors','Mean latency ms'],s1_table,
    [1.2,1.1,.75,1.0,1.85,.6],font=7.8)
add_caption(doc,'Table 2. Stage 1 generated-ID validation. MER and region errors are computed against the known synthetic QR matrix; latency includes model inference and ZXing-C++ decoding on CPU.')
add_figure(doc,FIGS/'stage1_validation_dr_mer.png',6.45,'Figure 1. Exact decode rate and module error rate by impairment category on the 200 new validation identities.')

heading(doc,'Stage 2 severity results')
doc.add_paragraph('L3 (shadow illumination factor 0.30) is the informative difficulty point: RAW succeeds on 17/55 images (30.9%), leaving both successful and failed cases. L4 (0.20) is an extreme floor for RAW at 2/55 (3.6%). At L1 and L2, RAW succeeds on 96.4% and 87.3%, respectively.')
main_methods=[('RAW','RAW'),('CLASSICAL_ORIG','Original classical'),('QR_AWARE','QR-aware classical'),('UNET_SINGLE_STANDARD','Fine-tuned SINGLE'),('UNET_ALL_STANDARD','Fine-tuned ALL'),('UNET_ALL_STRUCTURE_WEIGHTED','Fine-tuned ALL + QR loss'),('UNET_PRIOR_STANDARD_PILOT','Prior standard pilot'),('UNET_PRIOR_STRUCTURE_WEIGHTED_PILOT','Prior weighted pilot')]
stage2_table=[]
for m,label in main_methods:
    a=s2(m,3);b=s2(m,4)
    stage2_table.append((label,f"{a['exact_decodes']}/55 ({pct(a['exact_decode_rate'])}); {pct(a['exact_ci_identity_low'])}–{pct(a['exact_ci_identity_high'])}",
        f"{a['recoveries']}/{a['regressions']}",f"{float(a['mean_latency_ms']):.1f}",
        f"{b['exact_decodes']}/55 ({pct(b['exact_decode_rate'])}); {pct(b['exact_ci_identity_low'])}–{pct(b['exact_ci_identity_high'])}",
        f"{b['recoveries']}/{b['regressions']}",f"{float(b['mean_latency_ms']):.1f}"))
add_table(doc,['Method','L3 exact DR (identity 95% CI)','L3 rec / reg','L3 ms','L4 exact DR (identity 95% CI)','L4 rec / reg','L4 ms'],stage2_table,
    [1.3,1.38,.68,.48,1.38,.68,.48],font=7.25)
add_caption(doc,'Table 3. Stage 2 synthetic validation, 55 images per level. CIs resample the 10 QR identities. Mean latency includes restoration/model inference and exact decoding. All methods had zero false decodes.')
doc.add_paragraph('At L3, the classical methods recovered 28–30 of the RAW failures with no regression. Fine-tuned ALL and ALL + QR loss each recovered one image and regressed on 13 RAW successes; U-NET SINGLE had 15 regressions and no recovery. At L4, the fine-tuned U-Nets had zero net improvement, while the classical methods recovered 32–39 images without a measured regression. The preserved pilot checkpoints also exceeded the newly fine-tuned U-Nets at L3 and L4.')
add_figure(doc,FIGS/'stage2_decode_rate_by_severity.png',6.55,'Figure 2. Exact decode rate by severity for the common-ROI validation protocol; shaded bands show identity-bootstrap 95% intervals.')

heading(doc,'H1 combined-impairment training')
rows_h1=[]
for sev in range(1,5):
    r=hrow('H1_SINGLE_vs_ALL',sev)
    rows_h1.append((f'L{sev}',f"{pct(r['exact_rate_baseline'])} → {pct(r['exact_rate_comparison'])}",
                    f"{float(r['paired_rate_difference'])*100:+.1f} pp ({float(r['identity_bootstrap_95_low'])*100:+.1f} to {float(r['identity_bootstrap_95_high'])*100:+.1f})",
                    f"{r['recoveries']} / {r['regressions']}"))
add_table(doc,['Level','SINGLE → ALL DR','Paired Δ (95% CI)','Recovery / regression'],rows_h1,[.65,1.45,2.5,1.05],font=8.2)
doc.add_paragraph('H1 is partially supported at moderate severity. At L2, ALL improved aggregate U-Net DR from 27.3% to 52.7% (+25.5 percentage points; 95% identity-bootstrap CI +16 to +40; 19 recoveries, 5 regressions). For DS, DR moved from 9.1% to 81.8% (+72.7 points; CI +40 to +100); for SG, from 18.2% to 63.6% (+45.5 points; CI +15 to +75). However, at L3 the aggregate difference was +5.5 points (CI −4 to +16), and at L4 it was +3.6 points (CI 0 to +10). Both U-Net variants decoded 0/11 examples for each of S, DS, SG, and DSG at L3 and L4. The benefit is severity-limited and does not exceed the RAW or classical baselines.')

heading(doc,'H2 structure-weighted loss')
rows_h2=[]
for sev in range(1,5):
    r=hrow('H2_STANDARD_vs_STRUCTURE_WEIGHTED',sev)
    rows_h2.append((f'L{sev}',f"{pct(r['exact_rate_baseline'])} → {pct(r['exact_rate_comparison'])}",
                    f"{float(r['paired_rate_difference'])*100:+.1f} pp ({float(r['identity_bootstrap_95_low'])*100:+.1f} to {float(r['identity_bootstrap_95_high'])*100:+.1f})",
                    f"{r['recoveries']} / {r['regressions']}"))
add_table(doc,['Level','Standard → weighted DR','Paired Δ (95% CI)','Recovery / regression'],rows_h2,[.65,1.65,2.5,1.05],font=8.2)
doc.add_paragraph('H2 is not supported by decoder performance: the paired confidence intervals include zero at all levels. At L3 both models decoded 5/55, with four recoveries and four regressions between them; at L4 both decoded 2/55, with one recovery and one regression. Stage 1 MER was numerically lower under the weighted loss, but its validation set was already at a near-zero error floor. Stage 2 MER is not reported because the paired Clean-photo matrix is not an independent, pixel-aligned target.')
add_figure(doc,FIGS/'stage2_hypothesis_contrasts.png',6.55,'Figure 3. Paired H1 and H2 exact decode differences by condition on locked validation identities; shaded regions are identity-bootstrap 95% intervals.')

heading(doc,'Visual review of actual outputs')
doc.add_paragraph('The panels use unretouched evaluation outputs, resized only for display. Each row shows a paired Clean reference matrix, the synthetic shadow ROI, the original and QR-aware classical outputs, both fine-tuned U-Nets, the structure-weighted output, and exact decoder outcomes. The Clean matrix is a reference for viewing, not independent Stage 2 MER ground truth.')
add_figure(doc,FIGS/'shadow_examples.png',6.65,'Figure 4. Four actual outcome classes against U-NET ALL: RAW failure with recovery, unchanged success, regression, and failure. Exact outcomes for every displayed method are written in validation_example_outcomes.csv.')
doc.add_paragraph('In the regression example, RAW and both classical restorers decode, while the U-Nets do not; the predicted image changes the module boundaries enough to lose a previously valid decode. In the failure example, classical restoration recovers the symbol but the U-Nets fail, with the zoom showing disrupted finder/timing structure. The successful-recovery example shows that one L4 recovery can occur, but does not change the aggregate zero-net result at that severity. The unchanged-success example shows a case where the additional processing did not change decoder success.')
add_figure(doc,FIGS/'validation_module_zoom.png',6.4,'Figure 5. Nearest-neighbor zooms of the same examples, showing finder and timing pattern damage in actual model outputs.')

heading(doc,'Previous Artifact B results and protocol note')
doc.add_paragraph('The preserved pilot report audited six image archives (1,160 synthetic images), reported clean archive CRC checks and SHA-256 matches, and recorded L3 results of RAW 18/55, QR-aware classical 51/55, and RAW-first fallback 52/55. Those historical figures used the prior fixed-frame evaluation path. The current controlled U-Net comparison uses one input-detected, perspective-normalized 256 × 256 ROI for every method; its L3 values are RAW 17/55 and QR-aware classical 45/55. Because the ROI policy changed, these values are not a paired before/after comparison. The prior report and its tables remain unchanged in the package.')
doc.add_paragraph('The previous real-photo analysis reported RAW 108/110 and QR-aware 104/110 on the real held-out identities. Those identities had already been examined before this phase, so the old result is not an independent test estimate. No real held-out image was accessed in the present training, model selection, or validation run.')
doc.add_paragraph('Two exploratory runs are retained but excluded from the final H1/H2 claims. One used a regenerated QR reference whose version did not match the photographed physical code. A later corrected-target run used unmatched treatment seeds. The reported model results are from the final audited-reference, matched-seed training run.')

heading(doc,'Proposal requirements status')
heading(doc,'Complete for this phase',2)
for txt in [
    'Generated 4,000 unique Stage 1 training QR identities (2,000 per training arm) and 200 separate validation identities, with zero payload overlap across arms, validation, and protected prior validation/test payloads.',
    'Completed Stage 1 pretraining and Stage 2 fine-tuning checkpoints for SINGLE, ALL, and ALL + structure-weighted loss, with fixed budgets, training logs, validation histories, sample inputs/outputs, and loss curves.',
    'Completed validation-only H1/H2 comparisons, exact decode rates, paired recoveries/regressions, false-decode counts, latency, identity-cluster intervals, valid Stage 1 MER, and fixed/finder/timing/alignment error summaries.',
    'Created actual-output panels for all four paired decoder outcomes and a module/finder/timing zoom. Preserved the prior Artifact B project and checkpoints unchanged.'
]:bullet(doc,txt)
heading(doc,'Pending or limited',2)
for txt in [
    'Stage 2 used 693 of 940 source images from 36 of 45 training identities; 247 images and nine identities did not pass clean-reference or geometry checks. The fine-tune is complete on the audited subset; full use of all 45 identities remains pending if valid target geometry becomes available.',
    'Stage 2 MER and region-specific module errors remain unavailable without independent QR matrix annotations and trustworthy pixel alignment.',
    'Final held-out synthetic testing was intentionally not run. No independent final test claim is made.',
    'The remaining Proposal Table 1 systems outside the U-Net branches and the included RAW/classical comparisons are not evaluated in this Stage 1/2 package.',
    'External validity on newly captured physical photographs remains unmeasured. No physical images were requested or collected in this phase; any later study should use L3-like moderate shadows with L2 controls and L4 stress cases.'
]:bullet(doc,txt)

heading(doc,'Reproducibility and files')
doc.add_paragraph('The full package includes the training runner, analysis-figure source, JSON configuration, aligned data arrays and manifests, six final checkpoints, per-step logs, prediction-level outputs, aggregate result tables, audit records, actual visual examples, and the unchanged prior project. From the package root, the principal commands are:')
p=doc.add_paragraph();p.paragraph_format.left_indent=Inches(.25);p.paragraph_format.space_after=Pt(2)
for line in ['python experiment/src/stage1_stage2.py --stage stage1_data  # regenerate deterministic Stage 1 pairs if removed',
             'python experiment/src/stage1_stage2.py --stage train --stage1-steps 240 --stage2-steps 240',
             'python experiment/src/stage1_stage2.py --stage stage1_evaluate',
             'python experiment/src/stage1_stage2.py --stage evaluate',
             'python experiment/src/make_analysis_figures.py']:
    r=p.add_run(line+'\n');r.font.name='Consolas';r.font.size=Pt(8.5)
doc.add_paragraph('The editable report is a DOCX addendum for import into Google Docs. A Google Docs connection was not available in this workspace, so the prior report remains preserved and this new report provides the Stage 1/2 update.')

doc.core_properties.title='ECE 253 Artifact B Stage 1 and Stage 2 Progress Report'
doc.core_properties.subject='Synthetic QR shadow restoration pretraining and validation'
doc.core_properties.author='ECE 253 Artifact B research project'
doc.save(OUT)
print(OUT)
