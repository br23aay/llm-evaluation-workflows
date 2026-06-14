# Report Module
# Console summary, JSON export, and HTML report generators

import json
import os
from datetime import datetime, timezone
from typing import List


class ReportGenerator:
      def __init__(self, results):
                self.results = results
                self.generated_at = datetime.now(timezone.utc).isoformat()

      def console_summary(self):
                total = len(self.results)
                pass_count = sum(1 for r in self.results if r.status == "PASS")
                warn_count = sum(1 for r in self.results if r.status == "WARN")
                fail_count = sum(1 for r in self.results if r.status == "FAIL")
                avg_composite = sum(r.composite_score for r in self.results) / max(total, 1)
                avg_h = sum(r.hallucination_score for r in self.results) / max(total, 1)
                avg_s = sum(r.safety_score for r in self.results) / max(total, 1)
                avg_a = sum(r.alignment_score for r in self.results) / max(total, 1)

          print(f"\n{'='*60}")
        print(f"LLM Evaluation Summary — {self.generated_at[:10]}")
        print(f"{'='*60}")
        print(f"Total evaluated : {total}")
        print(f"PASS            : {pass_count}")
        print(f"WARN            : {warn_count}")
        print(f"FAIL            : {fail_count}")
        print(f"Avg composite   : {avg_composite:.3f}")
        print(f"Hallucination   : {avg_h:.3f}")
        print(f"Safety          : {avg_s:.3f}")
        print(f"Alignment       : {avg_a:.3f}")
        print(f"{'='*60}\n")

    def to_json(self, output_path: str = "results/eval_results.json"):
              os.makedirs(os.path.dirname(output_path), exist_ok=True)
              total = len(self.results)
              data = {
                  "generated_at": self.generated_at,
                  "total_evaluated": total,
                  "pass_count": sum(1 for r in self.results if r.status == "PASS"),
                  "warn_count": sum(1 for r in self.results if r.status == "WARN"),
                  "fail_count": sum(1 for r in self.results if r.status == "FAIL"),
                  "avg_composite": round(sum(r.composite_score for r in self.results) / max(total, 1), 4),
                  "results": [
                      {
                          "output_id": r.output_id,
                          "status": r.status,
                          "composite_score": r.composite_score,
                          "hallucination_score": r.hallucination_score,
                          "safety_score": r.safety_score,
                          "alignment_score": r.alignment_score,
                          "hallucination_verdict": r.hallucination_verdict,
                          "safety_verdict": r.safety_verdict,
                          "alignment_verdict": r.alignment_verdict,
                          "flags": r.flags,
                      }
                      for r in self.results
                  ],
              }
              with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        print(f"JSON report saved to {output_path}")
        return output_path

    def to_html(self, output_path: str = "results/eval_report.html"):
              os.makedirs(os.path.dirname(output_path), exist_ok=True)
        total = len(self.results)
        rows = "".join(
                      f"<tr><td>{r.output_id}</td><td>{r.status}</td>"
                      f"<td>{r.composite_score:.3f}</td>"
                      f"<td>{r.hallucination_score:.3f}</td>"
                      f"<td>{r.safety_score:.3f}</td>"
                      f"<td>{r.alignment_score:.3f}</td></tr>"
                      for r in self.results
        )
        html = f"""<!DOCTYPE html>
        <html lang=\"en\"><head><meta charset=\"UTF-8\">
        <title>LLM Evaluation Report</title></head>
        <body>
        <h1>LLM Evaluation Report</h1>
        <p>Generated: {self.generated_at} | Total: {total}</p>
        <table border=\"1\" cellpadding=\"6\">
        <tr><th>ID</th><th>Status</th><th>Composite</th>
        <th>Hallucination</th><th>Safety</th><th>Alignment</th></tr>
        {rows}
        </table>
        </body></html>"""
        with open(output_path, "w", encoding="utf-8") as f:
                      f.write(html)
                  print(f"HTML report saved to {output_path}")
        return output_path
