import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  BarChart3,
  Briefcase,
  Compass,
  LayoutDashboard,
  ListChecks,
  Sparkles,
  Star,
  Search,
} from "lucide-react";
import { Dashboard } from "@/components/mf/Dashboard";
import { Discovery, RiskBadge } from "@/components/mf/Discovery";
import { Questionnaire } from "@/components/mf/Questionnaire";
import { PredictionModal } from "@/components/mf/PredictionModal";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FUNDS, MODEL_STATS, type Fund } from "@/lib/fund-data";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "FundIQ — AI Mutual Fund Analytics & Recommendations" },
      {
        name: "description",
        content:
          "Explore 14,294 Indian mutual fund schemes with XGBoost 5-observation NAV predictions, explainable AI insights and a 4-layer recommendation engine.",
      },
      { property: "og:title", content: "FundIQ — AI Mutual Fund Analytics" },
      {
        property: "og:description",
        content:
          "XGBoost forward NAV predictions at 75.37% directional accuracy across 14,294 Indian mutual fund schemes.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const TABS = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "discovery", label: "Fund Discovery", icon: Compass },
  { key: "recommendation", label: "Questionnaire", icon: ListChecks },
  { key: "analytics", label: "Analytics", icon: BarChart3 },
  { key: "watchlist", label: "Watchlist", icon: Star },
  { key: "portfolio", label: "Portfolio", icon: Briefcase },
] as const;

type TabKey = (typeof TABS)[number]["key"];

function Index() {
  const [tab, setTab] = useState<TabKey>("dashboard");
  const [selected, setSelected] = useState<Fund | null>(null);
  const [watchlist, setWatchlist] = useState<string[]>([]);

  const toggleWatch = (f: Fund) =>
    setWatchlist((w) => (w.includes(f.code) ? w.filter((c) => c !== f.code) : [...w, f.code]));

  const watched = FUNDS.filter((f) => watchlist.includes(f.code));

  return (
    <div className="min-h-screen lg:flex">
      <aside className="glass sticky top-0 z-20 flex gap-2 overflow-x-auto p-3 lg:h-screen lg:w-64 lg:flex-col lg:gap-1 lg:p-5">
        <div className="mb-0 hidden items-center gap-2 lg:mb-6 lg:flex">
          <div className="grid size-9 place-items-center rounded-xl bg-primary/20 text-primary">
            <Sparkles className="size-5" />
          </div>
          <div>
            <p className="font-heading font-semibold">FundIQ</p>
            <p className="text-[11px] text-muted-foreground">AI Fund Intelligence</p>
          </div>
        </div>
        {TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`flex shrink-0 items-center gap-2 rounded-xl px-3 py-2 text-sm transition-colors ${
              tab === key
                ? "bg-primary/20 text-foreground"
                : "text-muted-foreground hover:bg-secondary/50"
            }`}
          >
            <Icon className="size-4" />
            {label}
          </button>
        ))}
      </aside>

      <main className="flex-1 px-4 py-6 lg:px-8">
        <header className="glass mb-6 flex flex-wrap items-center gap-4 rounded-2xl p-4">
          <div className="flex-1">
            <h1 className="text-gradient font-heading text-2xl font-semibold">
              Mutual Fund AI Analytics
            </h1>
            <p className="text-xs text-muted-foreground">
              XGBoost 5-observation forward NAV predictions · {MODEL_STATS.accuracy}% directional
              accuracy
            </p>
          </div>
          <Badge className="border-0 bg-bull/15 text-bull">
            <span className="live-dot mr-2 inline-block size-2 rounded-full bg-bull" />
            Markets live
          </Badge>
          <Button variant="secondary" size="sm" onClick={() => setTab("discovery")}>
            <Search className="size-4" /> Search schemes
          </Button>
          <div className="flex items-center gap-2 rounded-xl border border-border/60 bg-secondary/40 px-3 py-1.5">
            <div className="grid size-8 place-items-center rounded-full bg-violet/25 text-xs font-semibold text-violet">
              HS
            </div>
            <div className="text-xs">
              <p className="font-medium">Harish S</p>
              <p className="text-muted-foreground">Investor · 231001057</p>
            </div>
          </div>
        </header>

        {tab === "dashboard" && <Dashboard onAnalyse={setSelected} />}
        {(tab === "discovery" || tab === "analytics") && (
          <Discovery onAnalyse={setSelected} watchlist={watchlist} toggleWatch={toggleWatch} />
        )}
        {tab === "recommendation" && <Questionnaire onAnalyse={setSelected} />}
        {(tab === "watchlist" || tab === "portfolio") && (
          <div className="glass rounded-2xl p-6">
            <h2 className="font-heading text-lg font-semibold">
              {tab === "watchlist" ? "Your watchlist" : "Portfolio tracker"}
            </h2>
            {watched.length === 0 ? (
              <p className="mt-2 text-sm text-muted-foreground">
                No schemes saved yet — add funds from Fund Discovery.
              </p>
            ) : (
              <div className="mt-4 space-y-3">
                {watched.map((f) => (
                  <div
                    key={f.code}
                    className="flex flex-wrap items-center gap-3 rounded-xl border border-border/50 bg-secondary/25 p-4"
                  >
                    <div className="flex-1">
                      <p className="font-medium">{f.name}</p>
                      <p className="text-xs text-muted-foreground">
                        NAV ₹{f.nav.toFixed(2)} · Pred {f.predictedReturn.toFixed(2)}% · P(up){" "}
                        {(f.upProbability * 100).toFixed(0)}%
                      </p>
                    </div>
                    <RiskBadge vol={f.volatility20d} />
                    <Button size="sm" variant="secondary" onClick={() => setSelected(f)}>
                      Analytics
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => toggleWatch(f)}>
                      Remove
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      <PredictionModal fund={selected} onOpenChange={(o) => !o && setSelected(null)} />
    </div>
  );
}
