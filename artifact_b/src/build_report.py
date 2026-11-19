from pathlib import Path
import csv,json
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1]
def readcsv(path):
 with (ROOT/path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def shade(cell,fill):
 tcPr=cell._tc.get_or_add_tcPr();shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);tcPr.append(shd)
def borders(table):
 tblPr=table._tbl.tblPr;b=OxmlElement('w:tblBorders')
 for edge in ('top','left','bottom','right','insideH','insideV'):
  e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'5');e.set(qn('w:space'),'0');e.set(qn('w:color'),'D9D9D9');b.append(e)
 tblPr.append(b)
def set_cell(cell,text,bold=False,color='222222',size=8,align=None):
 cell.text='';p=cell.paragraphs[0]
 if align is not None:p.alignment=align
 p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0)
 r=p.add_run(str(text));r.bold=bold;r.font.name='Aptos';r.font.size=Pt(size);r.font.color.rgb=RGBColor.from_string(color)
 cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
 tc=cell._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
 for side,val in [('top','70'),('start','90'),('bottom','70'),('end','90')]:
  el=OxmlElement('w:'+side);el.set(qn('w:w'),val);el.set(qn('w:type'),'dxa');mar.append(el)
 tc.append(mar)
def table(doc,headers,rows,widths=None,small=8):
 t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 borders(t)
 for j,h in enumerate(headers):set_cell(t.rows[0].cells[j],h,True,'FFFFFF',small,WD_ALIGN_PARAGRAPH.CENTER);shade(t.rows[0].cells[j],'234B70')
 trPr=t.rows[0]._tr.get_or_add_trPr();rep=OxmlElement('w:tblHeader');rep.set(qn('w:val'),'true');trPr.append(rep)
 for i,row in enumerate(rows):
  cells=t.add_row().cells
  trPr=t.rows[-1]._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trPr.append(cant)
  for j,val in enumerate(row):
   set_cell(cells[j],val,False,'222222',small,WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT)
   if i%2:shade(cells[j],'F1F5F8')
 if widths:
  for rr in t.rows:
   for c,w in zip(rr.cells,widths):c.width=Inches(w)
 doc.add_paragraph().paragraph_format.space_after=Pt(2)
 return t
def addp(doc,text,style=None):
 p=doc.add_paragraph(style=style);p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.08;p.paragraph_format.keep_together=True;p.add_run(text);return p
def heading(doc,text,level=1):
 p=doc.add_paragraph(style=f'Heading {level}');p.paragraph_format.keep_with_next=True;p.add_run(text);return p
def image_page(doc,path,caption):
 s=doc.add_section(WD_SECTION.NEW_PAGE);s.orientation=1;s.page_width=Inches(11);s.page_height=Inches(8.5);s.top_margin=Inches(.45);s.bottom_margin=Inches(.45);s.left_margin=Inches(.55);s.right_margin=Inches(.55)
 p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(7);p.add_run().add_picture(str(ROOT/path),width=Inches(9.8))
 cp=doc.add_paragraph();cp.alignment=WD_ALIGN_PARAGRAPH.CENTER;cp.paragraph_format.keep_together=True;rr=cp.add_run(caption);rr.italic=True;rr.font.size=Pt(9)
def pct(v):return f'{100*float(v):.1f}%'
def main():
 out=ROOT/'ArtifactB_Stage2_Repair_Research_Addendum.docx';doc=Document();sec=doc.sections[0];sec.top_margin=Inches(.7);sec.bottom_margin=Inches(.7);sec.left_margin=Inches(.8);sec.right_margin=Inches(.8)
 styles=doc.styles
 for nm in ['Normal','Body Text']:
  styles[nm].font.name='Aptos';styles[nm]._element.rPr.rFonts.set(qn('w:eastAsia'),'Aptos');styles[nm].font.size=Pt(10.5);styles[nm].font.color.rgb=RGBColor(34,34,34)
 for nm,sz in [('Title',24),('Heading 1',16),('Heading 2',12)]:
  st=styles[nm];st.font.name='Aptos Display';st._element.rPr.rFonts.set(qn('w:eastAsia'),'Aptos Display');st.font.size=Pt(sz);st.font.bold=True;st.font.color.rgb=RGBColor(0,0,0)
  st.paragraph_format.space_before=Pt(11);st.paragraph_format.space_after=Pt(5)
  pPr=st._element.get_or_add_pPr();pbdr=pPr.find(qn('w:pBdr'))
  if pbdr is not None:pPr.remove(pbdr)
 title=doc.add_paragraph(style='Title');title.paragraph_format.space_after=Pt(3);title.add_run('Artifact B Stage 2 Scientific Repair')
 sub=doc.add_paragraph();sub.paragraph_format.space_after=Pt(13);r=sub.add_run('Checkpoint audit, supervision alignment, domain shift, and controlled validation');r.font.size=Pt(11);r.font.color.rgb=RGBColor(78,93,105)
 addp(doc,'Research update for UCSD ECE 253. This addendum preserves the historical Stage 1/2 work and reports a controlled repair of the Stage 2 U-Net experiment. No held-out test identities were accessed and Stage 3 was not run.')
 heading(doc,'Main finding')
 addp(doc,'The prior Stage 2 validation results reproduce, but their training summary lists step 120 while all three saved Stage 2 checkpoints and training logs select step 240. The original Stage 2 pixel targets were rendered from paired Clean QR matrices without an independently established pixel registration to the perspective-normalized input ROIs. This leaves the photographic pixel loss scientifically uncertain. A new matched experiment uses fully digital pairs with a shared known homography and the same QR identities across H1 arms. On the existing Stage 2 synthetic validation set, repaired ALL improves over RAW and the old U-Nets at L3/L4, but remains below classical methods. The structure-weighted loss does not improve severe-shadow decoding over the standard loss.')
 heading(doc,'1. Audit of the prior results')
 addp(doc,'The evaluation was reconstructed from the stored validation ROIs, the fixed ROI policy, and ZXing-C++ 3.1.1. One decoder result is compared after decoding to the recorded payload hash; the decoder itself receives no payload or target geometry. All 160 historical method × level × condition summary cells reproduce their stored exact-decode and false-decode counts.')
 rec=json.loads((ROOT/'audit/metadata_reconciliation.json').read_text())
 addp(doc,'Checkpoint reconciliation: all historical Stage 2 checkpoints record optimizer step 240, and the corresponding logs have their minimum validation loss at step 240. The corrected summary file instead says step 120. The checkpoint seed field stores the base data seed 253112026, whereas the run log records run seed 253112228. Original files were preserved; this report treats checkpoint metadata and the training log as the executed-run record.')
 ck=readcsv(Path('audit/checkpoint_manifest.csv'))
 ch=[]
 for r in ck:
  if 'STAGE2' in r.get('label',''):
   ch.append([r['label'],r.get('selected_step',r.get('step','')),r.get('validation_loss',''),r.get('sha256','')[:16]+'…'])
 if ch:table(doc,['Historical checkpoint','Selected step','Validation loss','SHA-256 prefix'],ch,[2.5,1,1.5,2.1],7.5)
 heading(doc,'2. Supervision alignment and domain shift')
 addp(doc,'The historical target-generation path reconstructed a QR matrix from a paired Clean photograph, checked decoded payload content, and rasterized an ideal QR on a 256 × 256 canvas. The degraded image ROI was independently detected and perspective-normalized. A matching payload does not establish pixel registration. We therefore withheld photographic-derived MER rather than treating payload verification as proof of module-boundary alignment.')
 addp(doc,'A detector-based corner residual proxy was measurable on a subset of input ROIs. Among ROIs where a second detector produced a quadrilateral, median residual was about 0.30 module at L1, 0.37 at L2, and roughly 2.4–2.5 modules at L3/L4. Severe shadows can impair the second detector, so this is evidence of alignment risk, not a definitive per-image registration estimate.')
 dg=readcsv(Path('audit/domain_gap_summary.csv'))
 addp(doc,'The Stage 1 generator trained only on QR module count 29 (Version 3); Stage 2 spans 25, 29, 33, 41, and 45 modules (Versions 2, 3, 4, 6, and 7). Stage 1 shadow factors ranged from 0.35 upward (minimum 0.35; 10th percentile about 0.42), with no 0.30 or 0.20 examples. Stage 2 uses factors 0.70, 0.50, 0.30, and 0.20, and photographic-derived ROIs have darker, lower-contrast, textured backgrounds. The earlier training distribution therefore underrepresented the most severe shadows and the QR/image statistics of the target domain.')
 heading(doc,'3. Controlled aligned repair experiment')
 cfg=json.loads((ROOT/'results/repair_training_config.json').read_text())
 addp(doc,'All repaired arms start from the same Stage 1 ALL standard checkpoint and use the same four-scale Compact U-Net (16/32/64/128 channels; 256 × 256 grayscale input), seed, optimizer, batch size, learning rate, and update budget. A pool of 37 matrix/capture units from 36 existing Stage 2 training identities produced 1,776 training pairs per arm. SINGLE uses S-only degradation; ALL uses balanced S/DS/SG/DSG with matched identity units and paired geometry/severity randomness. H2 uses the ALL input data and changes only the structure-weighted loss.')
 addp(doc,'The new 2,400-pair digital validation-selection set covers 200 existing Stage 1 synthetic validation identities, four conditions, and L2/L3/L4. Inputs and clean targets share the same recorded homography. Checkpoints were selected using this set; its results below are validation-selection results, not independent test evidence. Repaired checkpoints selected at step 160 for SINGLE standard and step 240 for ALL standard and ALL weighted.')
 train_rows=[]
 for name in ['repair_stage2_single','repair_stage2_all','repair_stage2_all_weighted']:
  rr=readcsv(Path(f'results/{name}_training_log.csv'))
  best=min(rr,key=lambda r:float(r['validation_loss']))
  train_rows.append([name.replace('repair_stage2_','').replace('_',' '),best['step'],f"{float(best['validation_loss']):.6f}",len(rr)])
 table(doc,['Repair arm','Selected step','Best validation loss','Logged evaluations'],train_rows,[2.3,1.1,1.7,1.7],8)
 heading(doc,'4. Existing Stage 2 synthetic validation results')
 addp(doc,'The existing Stage 2 validation split has 220 images and 10 QR identities; each severity level has 55 images. Invalid ROI cases remain failures in the denominator. Exact-decode rates use the same recorded payload hashes, fixed ROI, and decoder for all methods. Confidence intervals resample identities (5,000 bootstrap replicates).')
 ms=readcsv(Path('results/all_stage2_method_comparison.csv'))
 names=['RAW','CLASSICAL_ORIG','QR_AWARE','STAGE1_ALL_STANDARD','UNET_ALL_STANDARD','UNET_ALL_STRUCTURE_WEIGHTED','REPAIR_SINGLE','REPAIR_ALL','REPAIR_ALL_WEIGHTED']
 labels={'RAW':'RAW','CLASSICAL_ORIG':'Original classical','QR_AWARE':'QR-aware classical','STAGE1_ALL_STANDARD':'Stage 1 ALL standard','UNET_ALL_STANDARD':'Historical Stage 2 ALL standard','UNET_ALL_STRUCTURE_WEIGHTED':'Historical Stage 2 weighted','REPAIR_SINGLE':'Repair SINGLE standard','REPAIR_ALL':'Repair ALL standard','REPAIR_ALL_WEIGHTED':'Repair ALL weighted'}
 tab=[]
 for name in names:
  vals=[]
  for lev in (3,4):
   r=next(z for z in ms if z['method']==name and z['condition']=='ALL' and int(z['severity_level'])==lev)
   vals.append(f"{r['exact_decodes']}/55 ({pct(r['exact_decode_rate'])}; CI {pct(r['exact_ci_identity_low'])}–{pct(r['exact_ci_identity_high'])})")
  l3=next(z for z in ms if z['method']==name and z['condition']=='ALL' and int(z['severity_level'])==3)
  l4=next(z for z in ms if z['method']==name and z['condition']=='ALL' and int(z['severity_level'])==4)
  tab.append([labels[name],vals[0],f"{l3['recoveries']}/{l3['regressions']}",vals[1],f"{l4['recoveries']}/{l4['regressions']}",f"{float(l4['mean_latency_ms']):.1f}"])
 table(doc,['Method','L3 exact decode (identity CI)','L3 rec/reg','L4 exact decode (identity CI)','L4 rec/reg','Latency ms'],tab,[1.55,1.72,.72,1.72,.72,.65],7)
 addp(doc,'At L3, Repair ALL recovers 24 RAW failures and causes no regressions (41/55 versus RAW 17/55). At L4 it recovers 19 RAW failures with no regressions (21/55 versus RAW 2/55). Original classical remains higher at L3 (47/55), while QR-aware classical is higher at L4 (41/55). The repaired U-Net therefore improves over the historical U-Nets and RAW, but does not beat either classical method on this validation set. False decodes were zero for these methods.')
 heading(doc,'5. H1 and H2 paired comparisons')
 contrasts=readcsv(Path('results/repair_hypothesis_contrasts_stage2.csv'))
 ctab=[]
 for hyp,lev in [('H1_REPAIR_SINGLE_vs_ALL',3),('H1_REPAIR_SINGLE_vs_ALL',4),('H2_REPAIR_ALL_standard_vs_weighted',3),('H2_REPAIR_ALL_standard_vs_weighted',4)]:
  r=next(z for z in contrasts if z['hypothesis']==hyp and int(z['severity_level'])==lev and z['condition']=='ALL')
  ctab.append([('H1 SINGLE → ALL' if hyp.startswith('H1') else 'H2 standard → weighted'),f'L{lev}',f"{r['baseline_exact']}/{r['n_images']} → {r['comparison_exact']}/{r['n_images']}",f"{r['recoveries']}/{r['regressions']}",f"{int(r['net']):+d} ({float(r['paired_ci_low']):+.2f}, {float(r['paired_ci_high']):+.2f})"])
 table(doc,['Comparison','Level','Exact decodes','Paired rec/reg','Net change (95% identity CI)'],ctab,[1.9,.55,1.7,1.2,1.7],8)
 addp(doc,'ALL training outperforms SINGLE on the existing Stage 2 validation set at severe levels: the paired net difference is +15/55 at L3 (95% identity-bootstrap CI +0.10 to +0.44) and +12/55 at L4 (+0.08 to +0.32). The structure-weighted loss is not beneficial: weighted versus standard is −2 at L3 (CI −0.12 to +0.01) and −6 at L4 (CI −0.21 to −0.02). These contrasts are paired on the same identities and ROIs.')
 heading(doc,'6. Known-geometry digital validation and valid MER')
 dv=readcsv(Path('results/known_geometry_validation_comparison.csv'))
 dtab=[]
 for name in ['RAW','CLASSICAL_ORIG','QR_AWARE','STAGE1_ALL','STAGE1_WEIGHTED','STAGE2_ALL','STAGE2_WEIGHTED','REPAIR_SINGLE','REPAIR_ALL','REPAIR_ALL_WEIGHTED']:
  r=next(z for z in dv if z['method']==name and int(z['severity_level'])==4 and z['condition']=='ALL')
  dtab.append([name,f"{r['exact_decodes']}/{r['n_images']} ({pct(r['exact_decode_rate'])})",f"{r['recoveries']}/{r['regressions']}",f"{100*float(r['module_error_rate']):.3f}%",f"{100*float(r['finder_error_rate']):.2f}%",f"{100*float(r['timing_error_rate']):.2f}%",f"{float(r['mean_latency_ms']):.1f}"])
 table(doc,['Method at L4','Exact DR','Rec/reg','MER','Finder err','Timing err','ms'],dtab,[1.35,1.25,.8,.8,.8,.8,.55],7.3)
 addp(doc,'For this digital set, MER is valid because the QR matrices and the shared input/target homography are known independently. It samples nine interior locations per module and reports exact module agreement plus finder/timing/alignment regions. At L4, Repair ALL has 0.000% aggregate MER and 800/800 exact decodes; Repair SINGLE has 0.35% MER and 741/800 exact decodes. These unusually high scores belong to a finite synthetic validation-selection distribution and must not be generalized as independent-test performance. The historical Stage 2 photographic-derived ROI MER remains blank/withheld.')
 addp(doc,'False decodes were zero for every method across the 2,400 digital validation images. On the existing Stage 2 L3/L4 validation table, false decodes were also zero for all listed methods.')
 addp(doc,'The new digital validation latency uses amortized batched U-Net inference plus decoder time; the classical methods include measured restoration preprocessing and decoder time. Stage 2 validation latency is per-image inference/restoration plus decode. These timing setups differ, so use within-split comparisons and the detailed row-level CSVs rather than treating the two absolute latency columns as directly interchangeable.')
 heading(doc,'7. Visual failure analysis')
 addp(doc,'The panels show actual fixed-ROI outputs and exact decoder outcomes for a recovery, a historical Stage 2 U-Net regression, an unchanged success, and a case where all methods fail. A failed classical or neural threshold can distort finder borders or merge nearby modules; that changes the decoder’s localization and error-correction burden. The Clean matrix panel verifies content only and is explicitly not a pixel-registered MER target. Zooms use nearest-neighbor enlargement without retouching.')
 # Add figures as landscape pages for readable labels.
 heading(doc,'8. Proposal status and limitations')
 addp(doc,'Completed: checkpoint and metric audit; Stage 2 reproduction; historical and repaired model comparisons; alignment/domain-gap audits; matched H1/H2 training; exact decoding, recovery/regression, false decodes, latency, valid MER, identity-bootstrap intervals, and outcome-balanced visuals. Pending: untouched synthetic final test, Stage 3, and independent real-world validation. Earlier real-photo analysis examined the original test identities. The 2,400-image digital set selected checkpoints; it is not independent test data.')
 image_page(doc,Path('figures/stage2_repair_validation_rates.png'),'Figure 1. Exact decode rates and identity-bootstrap 95% intervals on the existing Stage 2 synthetic validation split.')
 image_page(doc,Path('figures/known_geometry_validation_rates.png'),'Figure 2. Exact decode rates on the known-geometry digital checkpoint-selection validation set. This is validation, not independent test data.')
 image_page(doc,Path('figures/stage2_repair_comparison.png'),'Figure 3. Original ROI, content-verified paired Clean matrix, baselines, historical models, and repaired models for four selected outcome classes. Pixel registration of the paired Clean matrix is unverified.')
 image_page(doc,Path('figures/stage2_module_failure_zoom.png'),'Figure 4. Nearest-neighbor crops of finder and horizontal timing-pattern regions from the same selected Stage 2 validation examples.')
 doc.core_properties.title='Artifact B Stage 2 Scientific Repair';doc.core_properties.subject='ECE 253 QR shadow restoration validation and model audit';doc.core_properties.author='ECE 253 Artifact B'
 doc.save(out);print(out)
if __name__=='__main__':main()
