import { useState } from "react";
import { ArrowLeft, ArrowRight, RotateCcw, Trophy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { recommend, type Answers, type Fund, type Goal, type Horizon, type RiskProfile } from "@/lib/fund-data";
import { RiskBadge } from "./Discovery";

const RISKS: { value: RiskProfile; hint: string }[] = [
  { value: "Low", hint: "Protect capital, minimal drawdowns" },
  { value: "Moderate", hint: "Balanced growth with some swings" },
  { value: "High", hint: "Maximise returns, accept volatility" },
];
const HORIZONS: { value: Horizon; label: string; hint: string }[] = [
  { value: "Short", label: "Short term", hint: "Under 1 year" },
  { value: "Medium", label: "Medium term", hint: "1 – 3 years" },
  { value: "Long", label: "Long term", hint: "Over 3 years" },
];
const GOALS: Goal[] = ["Wealth Creation", "Capital Preservation", "Tax Saving", "Income"];

export function Questionnaire({ onAnalyse }: { onAnalyse: (f: Fund) => void }) {
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<Partial<Answers>>({});

  const results = step === 3 && answers.risk && answers.horizon && answers.goal
    ? recommend(answers as Answers)
    : [];

  return (
    <div className="glass rounded-2xl p-6">
      <div className="mb-6 flex items-center gap-3">
        {["Risk profile", "Horizon", "Goal", "Recommendations"].map((label, i) => (
          <div key={label} className="flex flex-1 flex-col gap-2">
            <div
              className={`h-1.5 rounded-full ${i <= step ? "bg-primary" : "bg-secondary"}`}
            />
            <span
              className={`text-xs ${i <= step ? "text-foreground" : "text-muted-foreground"}`}
            >
              {i + 1}. {label}
            </span>
          </div>
        ))}
      </div>

      {step === 0 && (
        <Options
          title="How much risk are you comfortable with?"
          items={RISKS.map((r) => ({ key: r.value, label: r.value, hint: r.hint }))}
          selected={answers.risk}
          onSelect={(v) => {
            setAnswers((a) => ({ ...a, risk: v as RiskProfile }));
            setStep(1);
          }}
        />
      )}
      {step === 1 && (
        <Options
          title="How long do you plan to stay invested?"
          items={HORIZONS.map((h) => ({ key: h.value, label: h.label, hint: h.hint }))}
          selected={answers.horizon}
          onSelect={(v) => {
            setAnswers((a) => ({ ...a, horizon: v as Horizon }));
            setStep(2);
          }}
        />
      )}
      {step === 2 && (
        <Options
          title="What is your primary investment goal?"
          items={GOALS.map((g) => ({
            key: g,
            label: g,
            hint:
              g === "Tax Saving"
                ? "ELSS schemes with 80C benefit"
                : g === "Income"
                  ? "Regular payouts and stability"
                  : g === "Capital Preservation"
                    ? "Low drawdown, debt-tilted"
                    : "Long-run compounding",
          }))}
          selected={answers.goal}
          onSelect={(v) => {
            setAnswers((a) => ({ ...a, goal: v as Goal }));
            setStep(3);
          }}
        />
      )}

      {step === 3 && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Trophy className="size-5 text-amber" />
            <h3 className="font-heading text-lg font-semibold">Your matched schemes</h3>
            <Badge variant="outline">{answers.risk} risk</Badge>
            <Badge variant="outline">{answers.horizon} term</Badge>
            <Badge variant="outline">{answers.goal}</Badge>
          </div>
          {results.map((r, i) => (
            <div
              key={r.fund.code}
              className="glass-hover flex flex-col gap-3 rounded-xl border border-border/50 bg-secondary/25 p-4 md:flex-row md:items-center"
            >
              <div className="flex-1">
                <p className="font-medium">
                  <span className="mr-2 text-muted-foreground">#{i + 1}</span>
                  {r.fund.name}
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <Badge variant="outline">{r.fund.category}</Badge>
                  <RiskBadge vol={r.fund.volatility20d} />
                  <span>Pred. {r.fund.predictedReturn.toFixed(2)}%</span>
                  <span>P(up) {(r.fund.upProbability * 100).toFixed(0)}%</span>
                  <span>
                    Layers · cat {r.layers.category} · risk {r.layers.risk} · horizon{" "}
                    {r.layers.horizon} · ML {r.layers.ml}
                  </span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-muted-foreground">Suitability</p>
                <p className="font-heading text-2xl font-semibold text-bull">{r.score}</p>
              </div>
              <Button variant="secondary" onClick={() => onAnalyse(r.fund)}>
                View analytics
              </Button>
            </div>
          ))}
        </div>
      )}

      <div className="mt-6 flex gap-2">
        {step > 0 && (
          <Button variant="outline" onClick={() => setStep((s) => s - 1)}>
            <ArrowLeft className="size-4" /> Back
          </Button>
        )}
        {step === 3 ? (
          <Button
            variant="secondary"
            onClick={() => {
              setAnswers({});
              setStep(0);
            }}
          >
            <RotateCcw className="size-4" /> Start over
          </Button>
        ) : (
          <span className="flex items-center gap-1 text-xs text-muted-foreground">
            Pick an option to continue <ArrowRight className="size-3" />
          </span>
        )}
      </div>
    </div>
  );
}

function Options({
  title,
  items,
  selected,
  onSelect,
}: {
  title: string;
  items: { key: string; label: string; hint: string }[];
  selected?: string | undefined;
  onSelect: (v: string) => void;
}) {
  return (
    <div>
      <h3 className="mb-4 font-heading text-lg font-semibold">{title}</h3>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {items.map((it) => (
          <button
            key={it.key}
            onClick={() => onSelect(it.key)}
            className={`glass-hover rounded-xl border p-4 text-left ${
              selected === it.key
                ? "border-primary/60 bg-primary/15"
                : "border-border/60 bg-secondary/25"
            }`}
          >
            <p className="font-medium">{it.label}</p>
            <p className="mt-1 text-xs text-muted-foreground">{it.hint}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
