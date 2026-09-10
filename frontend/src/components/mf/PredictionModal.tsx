import { useMemo } from "react";
import {
  Area,
  ComposedChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertTriangle, Coins, Sparkles } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { featureImportance, navHistory, riskBand, type Fund } from "@/lib/fund-data";

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 12,
  color: "var(--popover-foreground)",
  fontSize: 12,
};

export function PredictionModal({
  fund,
  onOpenChange,
}: {
  fund: Fund | null;
  onOpenChange: (open: boolean) => void;
}) {
  const history = useMemo(() => (fund ? navHistory(fund) : []), [fund]);
  const features = useMemo(() => (fund ? featureImportance(fund) : []), [fund]);
  if (!fund) return null;

  const positive = fund.predictedReturn >= 0;

  return (
    <Dialog open={!!fund} onOpenChange={onOpenChange}>
      <DialogContent className="glass max-h-[92vh] overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle className="flex flex-wrap items-center gap-2 text-left text-xl">
            <Sparkles className="size-5 text-primary" />
            {fund.name}
          </DialogTitle>
        </DialogHeader>

        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="outline">{fund.category}</Badge>
          <Badge variant="outline">Code {fund.code}</Badge>
          <Badge variant="outline">{fund.isin}</Badge>
          <Badge variant="outline">Risk {riskBand(fund.volatility20d)}</Badge>
          {fund.zeroNavAnomaly && (
            <Badge className="border-0 bg-destructive/20 text-destructive">
              <AlertTriangle className="mr-1 size-3" /> Zero-NAV anomaly
            </Badge>
          )}
          {fund.recentIdcw && (
            <Badge className="border-0 bg-amber/20 text-amber">
              <Coins className="mr-1 size-3" /> Recent IDCW payout
            </Badge>
          )}
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <Metric
            label="Predicted 5-obs return"
            value={`${positive ? "+" : ""}${fund.predictedReturn.toFixed(2)}%`}
            tone={positive ? "bull" : "bear"}
          />
          <Metric label="Upward probability" value={`${(fund.upProbability * 100).toFixed(1)}%`} />
          <Metric label="Current NAV" value={`₹${fund.nav.toFixed(4)}`} />
        </div>

        <section className="rounded-xl border border-border/60 p-4">
          <h3 className="mb-3 text-sm font-semibold">Historical NAV · 20D & 60D moving averages</h3>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={history}>
                <defs>
                  <linearGradient id="navFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.55} />
                    <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis dataKey="day" hide />
                <YAxis
                  width={54}
                  domain={["auto", "auto"]}
                  tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Area
                  dataKey="nav"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
                  fill="url(#navFill)"
                  name="NAV"
                />
                <Line type="monotone" dataKey="ma20" stroke="var(--chart-2)" dot={false} strokeWidth={1.5} name="MA 20" />
                <Line
                  type="monotone"
                  dataKey="ma60"
                  stroke="var(--chart-3)"
                  dot={false}
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  name="MA 60"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-xl border border-border/60 p-4">
          <h3 className="mb-3 text-sm font-semibold">
            Explainable AI · feature attribution for this scheme
          </h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={features} layout="vertical" margin={{ left: 24 }}>
                <CartesianGrid stroke="var(--border)" horizontal={false} />
                <XAxis type="number" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={110}
                  tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--accent)" }} />
                <Bar dataKey="weight" fill="var(--chart-3)" radius={[0, 6, 6, 0]} name="Weight %" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </DialogContent>
    </Dialog>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "bull" | "bear";
}) {
  return (
    <div className="rounded-xl border border-border/60 bg-secondary/40 p-4">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p
        className={`font-heading text-2xl font-semibold ${
          tone === "bull" ? "text-bull" : tone === "bear" ? "text-bear" : "text-foreground"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
