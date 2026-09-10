import { useEffect, useMemo, useState } from "react";
import { LineChart, Search, Star } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FUNDS, riskBand, type Category, type Fund } from "@/lib/fund-data";

const CATEGORIES: (Category | "All")[] = ["All", "Equity", "Debt", "Hybrid", "Other"];

const SORTS = {
  pred: { label: "Predicted 5-obs return ↓", fn: (a: Fund, b: Fund) => b.predictedReturn - a.predictedReturn },
  prob: { label: "Upward probability ↓", fn: (a: Fund, b: Fund) => b.upProbability - a.upProbability },
  vol: { label: "20D volatility ↑", fn: (a: Fund, b: Fund) => a.volatility20d - b.volatility20d },
  dd: { label: "60D max drawdown ↑", fn: (a: Fund, b: Fund) => b.drawdown60d - a.drawdown60d },
} as const;

export function Discovery({
  onAnalyse,
  watchlist,
  toggleWatch,
}: {
  onAnalyse: (f: Fund) => void;
  watchlist: string[];
  toggleWatch: (f: Fund) => void;
}) {
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [category, setCategory] = useState<Category | "All">("All");
  const [sort, setSort] = useState<keyof typeof SORTS>("pred");
  const [limit, setLimit] = useState(25);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim().toLowerCase()), 250);
    return () => clearTimeout(t);
  }, [query]);

  const rows = useMemo(() => {
    return FUNDS.filter(
      (f) =>
        (category === "All" || f.category === category) &&
        (!debounced ||
          f.name.toLowerCase().includes(debounced) ||
          f.isin.toLowerCase().includes(debounced) ||
          f.code.includes(debounced)),
    ).sort(SORTS[sort].fn);
  }, [debounced, category, sort]);

  return (
    <div className="space-y-5">
      <div className="glass flex flex-col gap-4 rounded-2xl p-4 md:flex-row md:items-center">
        <div className="relative flex-1">
          <Search className="absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setLimit(25);
            }}
            placeholder="Search by scheme name, ISIN or scheme code…"
            className="border-border/60 bg-secondary/40 pl-9"
          />
        </div>
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as keyof typeof SORTS)}
          className="h-9 rounded-md border border-border/60 bg-secondary/40 px-3 text-sm"
        >
          {Object.entries(SORTS).map(([k, v]) => (
            <option key={k} value={k} className="bg-popover">
              {v.label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-wrap gap-2">
        {CATEGORIES.map((c) => (
          <button
            key={c}
            onClick={() => {
              setCategory(c);
              setLimit(25);
            }}
            className={`rounded-full border px-4 py-1.5 text-sm transition-colors ${
              category === c
                ? "border-primary/50 bg-primary/20 text-foreground"
                : "border-border/60 bg-secondary/30 text-muted-foreground hover:bg-secondary/60"
            }`}
          >
            {c}
          </button>
        ))}
        <span className="ml-auto self-center text-xs text-muted-foreground">
          {rows.length.toLocaleString("en-IN")} schemes matched
        </span>
      </div>

      <div className="glass overflow-hidden rounded-2xl">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] text-sm">
            <thead className="bg-secondary/40 text-xs tracking-wide text-muted-foreground uppercase">
              <tr>
                <th className="px-4 py-3 text-left">Scheme</th>
                <th className="px-4 py-3 text-right">NAV</th>
                <th className="px-4 py-3 text-right">Pred. 5-obs</th>
                <th className="px-4 py-3 text-left">Upward probability</th>
                <th className="px-4 py-3 text-right">Vol 20D</th>
                <th className="px-4 py-3 text-center">Risk</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, limit).map((f) => {
                const up = f.predictedReturn >= 0;
                return (
                  <tr key={f.code} className="border-t border-border/40 hover:bg-secondary/30">
                    <td className="px-4 py-3">
                      <p className="font-medium">{f.name}</p>
                      <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                        <Badge variant="outline" className="text-[10px]">
                          {f.amc}
                        </Badge>
                        <span>{f.isin}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">₹{f.nav.toFixed(2)}</td>
                    <td
                      className={`px-4 py-3 text-right font-semibold tabular-nums ${up ? "text-bull" : "text-bear"}`}
                    >
                      {up ? "+" : ""}
                      {f.predictedReturn.toFixed(2)}%
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-28 overflow-hidden rounded-full bg-secondary">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${f.upProbability * 100}%` }}
                          />
                        </div>
                        <span className="text-xs tabular-nums text-muted-foreground">
                          {(f.upProbability * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">{f.volatility20d.toFixed(2)}%</td>
                    <td className="px-4 py-3 text-center">
                      <RiskBadge vol={f.volatility20d} />
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex justify-end gap-2">
                        <Button size="sm" variant="secondary" onClick={() => onAnalyse(f)}>
                          <LineChart className="size-4" /> Analytics
                        </Button>
                        <Button
                          size="sm"
                          variant={watchlist.includes(f.code) ? "default" : "outline"}
                          onClick={() => toggleWatch(f)}
                        >
                          <Star className="size-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {limit < rows.length && (
          <div className="border-t border-border/40 p-4 text-center">
            <Button variant="secondary" onClick={() => setLimit((l) => l + 25)}>
              Load more schemes
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

export function RiskBadge({ vol }: { vol: number }) {
  const band = riskBand(vol);
  const cls =
    band === "Low"
      ? "bg-bull/15 text-bull"
      : band === "Moderate"
        ? "bg-amber/15 text-amber"
        : "bg-bear/15 text-bear";
  return <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${cls}`}>{band}</span>;
}
