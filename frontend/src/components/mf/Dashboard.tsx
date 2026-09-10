import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, Database, Gauge, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FUNDS, MODEL_STATS, type Category, type Fund } from "@/lib/fund-data";
import { RiskBadge } from "./Discovery";

const CATS: Category[] = ["Equity", "Debt", "Hybrid", "Other"];
const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];

export function Dashboard({ onAnalyse }: { onAnalyse: (f: Fund) => void }) {
  const gainers = useMemo(
    () => [...FUNDS].sort((a, b) => b.predictedReturn - a.predictedReturn).slice(0, 6),
    [],
  );

  const heat = useMemo(
    () =>
      CATS.map((c) => {
        const set = FUNDS.filter((f) => f.category === c);
        return {
          category: c,
          volatility: +(set.reduce((a, b) => a + b.volatility20d, 0) / (set.length || 1)).toFixed(2),
          count: set.length,
        };
      }),
    [],
  );

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat
          icon={<Gauge className="size-4" />}
          label="Model directional accuracy"
          value={`${MODEL_STATS.accuracy}%`}
          note="XGBoost · 5-observation forward"
          tone="bull"
        />
        <Stat
          icon={<Activity className="size-4" />}
          label="Debt fund accuracy"
          value={`${MODEL_STATS.debtAccuracy}%`}
          note="Highest confidence segment"
          tone="violet"
        />
        <Stat
          icon={<Database className="size-4" />}
          label="Tracked schemes"
          value={MODEL_STATS.totalSchemes.toLocaleString("en-IN")}
          note="All Indian AMCs"
        />
        <Stat
          icon={<TrendingUp className="size-4" />}
          label="Latest data timestamp"
          value={MODEL_STATS.dataAsOf}
          note="AMFI NAV feed synced"
          tone="amber"
        />
      </div>

      <section>
        <h2 className="mb-3 font-heading text-lg font-semibold">Top predicted gainers</h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {gainers.map((f) => (
            <div key={f.code} className="glass glass-hover rounded-2xl p-4">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="font-medium leading-snug">{f.name}</p>
                  <Badge variant="outline" className="mt-2 text-[10px]">
                    {f.category} · {f.amc}
                  </Badge>
                </div>
                <span className="font-heading text-xl font-semibold text-bull">
                  +{f.predictedReturn.toFixed(2)}%
                </span>
              </div>
              <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
                <span>NAV ₹{f.nav.toFixed(2)}</span>
                <span>P(up) {(f.upProbability * 100).toFixed(0)}%</span>
                <RiskBadge vol={f.volatility20d} />
              </div>
              <Button
                variant="secondary"
                size="sm"
                className="mt-4 w-full"
                onClick={() => onAnalyse(f)}
              >
                Open AI prediction
              </Button>
            </div>
          ))}
        </div>
      </section>

      <section className="glass rounded-2xl p-5">
        <h2 className="mb-1 font-heading text-lg font-semibold">Market risk heatmap</h2>
        <p className="mb-4 text-xs text-muted-foreground">
          Average 20-day volatility by asset class
        </p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={heat}>
              <CartesianGrid stroke="var(--border)" vertical={false} />
              <XAxis dataKey="category" tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
              <YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
              <Tooltip
                cursor={{ fill: "var(--accent)" }}
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 12,
                  fontSize: 12,
                }}
              />
              <Bar dataKey="volatility" radius={[8, 8, 0, 0]} name="Avg vol 20D %">
                {heat.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}

function Stat({
  icon,
  label,
  value,
  note,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  note: string;
  tone?: "bull" | "violet" | "amber";
}) {
  const color =
    tone === "bull" ? "text-bull" : tone === "violet" ? "text-violet" : tone === "amber" ? "text-amber" : "text-primary";
  return (
    <div className="glass glass-hover rounded-2xl p-5">
      <div className={`flex items-center gap-2 text-xs ${color}`}>
        {icon}
        <span className="text-muted-foreground">{label}</span>
      </div>
      <p className={`mt-3 font-heading text-3xl font-semibold ${color}`}>{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{note}</p>
    </div>
  );
}
