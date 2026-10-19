import csv, json, argparse, statistics
from pathlib import Path

def num(v):
    try:
        if v is None or str(v).strip()=='' : return None
        n=float(str(v).strip())
        return int(n) if n in (0,1) else n
    except Exception: return None

def main():
    ap=argparse.ArgumentParser(description='Score human-reviewed offline PDF QA evaluation CSV')
    ap.add_argument('--gold',default='qa_gold_answers_v3_115.csv')
    ap.add_argument('--results',default='qa_results_template_v3_115.csv')
    ap.add_argument('--out',default='evaluation_summary.json')
    a=ap.parse_args()
    with open(a.gold,encoding='utf-8-sig',newline='') as f: gold={r['qa_id']:r for r in csv.DictReader(f)}
    with open(a.results,encoding='utf-8-sig',newline='') as f: results=list(csv.DictReader(f))
    evaluated=[r for r in results if (r.get('system_answer') or '').strip()]
    metrics=['answer_correct_0_1','answer_complete_0_1','relevant_0_1','faithful_to_pdf_0_1','citation_correct_0_1','abstention_correct_0_1']
    vals={m:[] for m in metrics}; accepted=[]; by_type={}; latencies=[]
    for r in evaluated:
        g=gold.get(r.get('qa_id',''),{})
        answerable=str(g.get('answerable','')).lower() in ('true','1','yes')
        m={k:num(r.get(k)) for k in metrics}
        for k in metrics:
            if m[k] is not None and (k!='abstention_correct_0_1' or not answerable): vals[k].append(m[k])
        if answerable:
            required=[m[k] for k in metrics[:5]]
            passed=all(x==1 for x in required)
        else:
            passed=(m['abstention_correct_0_1']==1)
        accepted.append(int(passed))
        typ=g.get('question_type','unknown')
        by_type.setdefault(typ,[]).append(int(passed))
        latency=num(r.get('latency_seconds'))
        if latency is not None: latencies.append(float(latency))
    n_total=len(gold); n_eval=len(evaluated)
    report={
      'dataset_items':n_total,'evaluated_items':n_eval,
      'test_coverage_percent':round(100*n_eval/n_total,2) if n_total else 0,
      'accepted_answers':sum(accepted),'accepted_answer_rate_percent':round(100*sum(accepted)/n_eval,2) if n_eval else None,
      'target_80_percent_met':bool(100*sum(accepted)/n_eval>=80) if n_eval else False,
      'component_rates_percent':{k:round(100*sum(v)/len(v),2) if v else None for k,v in vals.items()},
      'accepted_rate_by_question_type_percent':{k:round(100*sum(v)/len(v),2) for k,v in by_type.items()},
      'latency_seconds':{'count':len(latencies),'mean':round(statistics.mean(latencies),3) if latencies else None,'median':round(statistics.median(latencies),3) if latencies else None,'max':round(max(latencies),3) if latencies else None},
      'method_note':'Human-judged pass rate; answerable items require correctness, completeness, relevance, faithfulness, and citation labels all equal 1. Unanswerable items require abstention_correct_0_1=1. Blank system_answer is treated as not evaluated, not as correct.'
    }
    Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2)); print('Report saved:',a.out)
if __name__=='__main__': main()
