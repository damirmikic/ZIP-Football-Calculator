/**
 * ZIP Football Calculator
 * * Implements Zero-Inflated Poisson distribution for football score modelling.
 * Features:
 * - Conversion between xG inputs and Supremacy/Expectancy
 * - Zero-Inflation parameter handling
 * - Market derivation (1X2, O/U, BTTS, Correct Score)
 * - Multi-clickable grid (Dutching)
 * - Independent Margins for Periods
 */

// --- Constants & Config ---
const MAX_GOALS = 7; // Grid size 0-7
const STANDARD_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];

// Store calculation results globally
let appState = {
    jointFull: null,
    jointH1: null,
    jointH2: null,
    inputs: null,
    htftData: null,
    scoresByGoals: null,
    dcGoals: null,
    result1X2Goals: null
};

// Store current selections for each grid: 'period' -> Set of "h-a" strings
let selections = {
    full: new Set(),
    h1: new Set(),
    h2: new Set()
};

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    calculateAndRender();
});

// --- Event Listeners ---
function initEventListeners() {
    // Input Mode Toggle
    document.getElementsByName('inputMode').forEach(radio => {
        radio.addEventListener('change', (e) => toggleInputMode(e.target.value));
    });

    // ZIP Toggle
    document.getElementById('useZip').addEventListener('change', (e) => toggleZipInputs(e.target.checked));

    // Calculate Button
    document.getElementById('calcBtn').addEventListener('click', calculateAndRender);

    // Margin Changes (Re-render odds without full recalc)
    ['marginFull', 'marginH1', 'marginH2'].forEach(id => {
        document.getElementById(id).addEventListener('input', () => {
            if(appState.jointFull) {
                // Update inputs state
                appState.inputs = getInputs(); 
                if(appState.inputs.isValid) renderAllMarkets();
            }
        });
    });

    // Tabs
    document.querySelectorAll('.tab-btn').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            e.target.classList.add('active');
            document.getElementById(`tab-${e.target.dataset.tab}`).classList.add('active');
        });
    });
}

function toggleInputMode(mode) {
    if (mode === 'xg') {
        document.getElementById('modeA-container').classList.add('active');
        document.getElementById('modeB-container').classList.remove('active');
    } else {
        document.getElementById('modeA-container').classList.remove('active');
        document.getElementById('modeB-container').classList.add('active');
    }
}

function toggleZipInputs(enabled) {
    const ids = ['piHome', 'piAway'];
    const container = document.getElementById('zip-inputs');
    if (enabled) {
        container.classList.remove('disabled');
        ids.forEach(id => document.getElementById(id).disabled = false);
    } else {
        container.classList.add('disabled');
        ids.forEach(id => document.getElementById(id).disabled = true);
    }
}

// --- Input Handling & Validation ---

function getInputs() {
    const mode = document.querySelector('input[name="inputMode"]:checked').value;
    const useZip = document.getElementById('useZip').checked;
    
    let data = {
        homeTeam: document.getElementById('homeTeam').value || 'Home',
        awayTeam: document.getElementById('awayTeam').value || 'Away',
        lambdaHomeFull: 0,
        lambdaAwayFull: 0,
        piHome: 0,
        piAway: 0,
        halfFactor: parseFloat(document.getElementById('halfFactor').value),
        marginFull: parseFloat(document.getElementById('marginFull').value) || 0,
        marginH1: parseFloat(document.getElementById('marginH1').value) || 0,
        marginH2: parseFloat(document.getElementById('marginH2').value) || 0,
        isValid: true,
        errorMsg: ''
    };

    // Validate inputs
    if (mode === 'xg') {
        data.lambdaHomeFull = parseFloat(document.getElementById('xgHome').value);
        data.lambdaAwayFull = parseFloat(document.getElementById('xgAway').value);
        if (data.lambdaHomeFull < 0 || data.lambdaAwayFull < 0) {
            data.isValid = false; data.errorMsg = "xG values must be non-negative.";
        }
    } else {
        const sup = parseFloat(document.getElementById('supremacy').value);
        const exp = parseFloat(document.getElementById('expectancy').value);
        if (exp <= 0) {
            data.isValid = false; data.errorMsg = "Total expectancy must be > 0.";
        } else {
            data.lambdaHomeFull = (exp + sup) / 2;
            data.lambdaAwayFull = (exp - sup) / 2;
            if (data.lambdaHomeFull < 0 || data.lambdaAwayFull < 0) {
                data.isValid = false; data.errorMsg = "Implied team goals negative.";
            }
        }
    }

    if (useZip) {
        data.piHome = parseFloat(document.getElementById('piHome').value);
        data.piAway = parseFloat(document.getElementById('piAway').value);
        if (data.piHome < 0 || data.piHome > 1 || data.piAway < 0 || data.piAway > 1) {
            data.isValid = false; data.errorMsg = "Pi must be 0-1.";
        }
    }
    
    // Safety clamp on margins
    if (data.marginFull < 0) data.marginFull = 0;
    if (data.marginH1 < 0) data.marginH1 = 0;
    if (data.marginH2 < 0) data.marginH2 = 0;

    return data;
}

// --- Math Model ---

const factorialCache = [1, 1, 2, 6, 24, 120, 720, 5040];
function factorial(n) {
    if (n < 0) return 1;
    if (n < factorialCache.length) return factorialCache[n];
    let res = factorialCache[factorialCache.length - 1];
    for (let i = factorialCache.length; i <= n; i++) res *= i;
    return res;
}

function computePoissonPMF(lambda, maxGoals, pi) {
    let probs = [];
    let sum = 0;
    const expLambda = Math.exp(-lambda);

    for (let k = 0; k <= maxGoals; k++) {
        let pStandard = (expLambda * Math.pow(lambda, k)) / factorial(k);
        let pZip = k === 0 ? pi + (1 - pi) * pStandard : (1 - pi) * pStandard;
        probs.push(pZip);
        sum += pZip;
    }
    return { probs, tail: 1 - sum };
}

function computeJointDistribution(lambdaH, lambdaA, piH, piA, maxGoals) {
    const distH = computePoissonPMF(lambdaH, maxGoals, piH);
    const distA = computePoissonPMF(lambdaA, maxGoals, piA);

    let matrix = [];
    let totalProb = 0;

    for (let i = 0; i <= maxGoals; i++) {
        matrix[i] = [];
        for (let j = 0; j <= maxGoals; j++) {
            let jointP = distH.probs[i] * distA.probs[j];
            matrix[i][j] = jointP;
            totalProb += jointP;
        }
    }
    return { matrix, tailProb: 1 - totalProb };
}

// --- Calculations ---

function calculateAndRender() {
    const inputs = getInputs();
    const errorBox = document.getElementById('error-box');

    if (!inputs.isValid) {
        errorBox.textContent = inputs.errorMsg;
        errorBox.classList.remove('hidden');
        return;
    }
    errorBox.classList.add('hidden');
    appState.inputs = inputs;

    // 1. Full Time
    appState.jointFull = computeJointDistribution(inputs.lambdaHomeFull, inputs.lambdaAwayFull, inputs.piHome, inputs.piAway, MAX_GOALS);

    // 2. 1st Half (Inputs Factor)
    // Common default 0.45
    const factorH1 = inputs.halfFactor; 
    appState.jointH1 = computeJointDistribution(inputs.lambdaHomeFull * factorH1, inputs.lambdaAwayFull * factorH1, inputs.piHome, inputs.piAway, MAX_GOALS);

    // 3. 2nd Half (Remainder)
    // 2nd half usually higher scoring. 1 - 0.45 = 0.55
    const factorH2 = 1.0 - factorH1;
    appState.jointH2 = computeJointDistribution(inputs.lambdaHomeFull * factorH2, inputs.lambdaAwayFull * factorH2, inputs.piHome, inputs.piAway, MAX_GOALS);

    renderAllMarkets();
}

// --- Market Derivation ---

function deriveMarkets(jointDist) {
    const matrix = jointDist.matrix;
    const size = matrix.length;
    let m = {
        homeWin: 0, draw: 0, awayWin: 0,
        bttsYes: 0, bttsNo: 0,
        winToNilHome: 0, winToNilAway: 0,
        cleanSheetHome: 0, cleanSheetAway: 0,
        overs: {}, exactTotals: [],
        map: {} // Map "h-a" -> prob for quick access
    };

    let maxTotal = (size - 1) * 2;
    for(let t=0; t<=maxTotal; t++) m.exactTotals[t] = 0;

    for (let h = 0; h < size; h++) {
        for (let a = 0; a < size; a++) {
            const p = matrix[h][a];
            m.map[`${h}-${a}`] = p;

            if (h > a) m.homeWin += p;
            else if (h === a) m.draw += p;
            else m.awayWin += p;

            if (h > 0 && a > 0) m.bttsYes += p;
            if (h > 0 && a === 0) m.winToNilHome += p;
            if (a > 0 && h === 0) m.winToNilAway += p;
            if (a === 0) m.cleanSheetHome += p;
            if (h === 0) m.cleanSheetAway += p;

            const total = h + a;
            if (total <= maxTotal) m.exactTotals[total] += p;

            STANDARD_LINES.forEach(line => {
                if (total > line) m.overs[line] = (m.overs[line] || 0) + p;
            });
        }
    }
    m.bttsNo = 1 - m.bttsYes;

    // Normalization for 1X2 (ignoring tail)
    const sum1x2 = m.homeWin + m.draw + m.awayWin;
    if (sum1x2 > 0) {
        m.homeWin /= sum1x2; m.draw /= sum1x2; m.awayWin /= sum1x2;
    }
    return m;
}

// Calculate Halftime/Full Time markets
function deriveHalfTimeFullTime(jointH1, jointH2) {
    const matrixH1 = jointH1.matrix;
    const matrixH2 = jointH2.matrix;
    const size = matrixH1.length;

    // 9 possible outcomes: HT result / FT result
    let htft = {
        'H-H': 0, 'H-D': 0, 'H-A': 0,
        'D-H': 0, 'D-D': 0, 'D-A': 0,
        'A-H': 0, 'A-D': 0, 'A-A': 0
    };

    // Also track HT/FT with goal totals for combination markets
    let htftGoals = {};
    STANDARD_LINES.forEach(line => {
        htftGoals[line] = {};
        Object.keys(htft).forEach(key => {
            htftGoals[line][key] = { over: 0, under: 0 };
        });
    });

    // Iterate through all combinations of HT and 2H scores
    for (let h1_h = 0; h1_h < size; h1_h++) {
        for (let h1_a = 0; h1_a < size; h1_a++) {
            const pHT = matrixH1[h1_h][h1_a];
            if (pHT === 0) continue;

            // Determine HT result
            let htResult = h1_h > h1_a ? 'H' : h1_h === h1_a ? 'D' : 'A';

            for (let h2_h = 0; h2_h < size; h2_h++) {
                for (let h2_a = 0; h2_a < size; h2_a++) {
                    const p2H = matrixH2[h2_h][h2_a];
                    if (p2H === 0) continue;

                    // Calculate full-time score
                    const ft_h = h1_h + h2_h;
                    const ft_a = h1_a + h2_a;
                    const ftTotal = ft_h + ft_a;

                    // Determine FT result
                    let ftResult = ft_h > ft_a ? 'H' : ft_h === ft_a ? 'D' : 'A';

                    // Combined probability (assuming independence)
                    const prob = pHT * p2H;

                    // Add to HT/FT market
                    const key = `${htResult}-${ftResult}`;
                    htft[key] += prob;

                    // Add to HT/FT + Goals markets
                    STANDARD_LINES.forEach(line => {
                        if (ftTotal > line) {
                            htftGoals[line][key].over += prob;
                        } else {
                            htftGoals[line][key].under += prob;
                        }
                    });
                }
            }
        }
    }

    return { htft, htftGoals };
}

// Calculate Final Score grouped by total goals
function deriveFinalScoreByGoals(jointDist) {
    const matrix = jointDist.matrix;
    const size = matrix.length;

    let scoresByGoals = {};
    for (let t = 0; t <= (size - 1) * 2; t++) {
        scoresByGoals[t] = [];
    }

    for (let h = 0; h < size; h++) {
        for (let a = 0; a < size; a++) {
            const p = matrix[h][a];
            const total = h + a;
            scoresByGoals[total].push({
                score: `${h}-${a}`,
                prob: p
            });
        }
    }

    // Sort each group by probability descending
    Object.keys(scoresByGoals).forEach(t => {
        scoresByGoals[t].sort((a, b) => b.prob - a.prob);
    });

    return scoresByGoals;
}

// Calculate Double Chance + Goals combinations
function deriveDoubleChanceGoals(markets) {
    let dcGoals = {};

    STANDARD_LINES.forEach(line => {
        const pOver = markets.overs[line] || 0;
        const pUnder = 1 - pOver;

        dcGoals[line] = {
            '1X': { over: 0, under: 0 },  // Home or Draw
            'X2': { over: 0, under: 0 },  // Draw or Away
            '12': { over: 0, under: 0 }   // Home or Away
        };
    });

    const matrix = appState.jointFull.matrix;
    const size = matrix.length;

    for (let h = 0; h < size; h++) {
        for (let a = 0; a < size; a++) {
            const p = matrix[h][a];
            const total = h + a;
            const isHome = h > a;
            const isDraw = h === a;
            const isAway = h < a;

            STANDARD_LINES.forEach(line => {
                const isOver = total > line;

                if (isHome || isDraw) {
                    dcGoals[line]['1X'][isOver ? 'over' : 'under'] += p;
                }
                if (isDraw || isAway) {
                    dcGoals[line]['X2'][isOver ? 'over' : 'under'] += p;
                }
                if (isHome || isAway) {
                    dcGoals[line]['12'][isOver ? 'over' : 'under'] += p;
                }
            });
        }
    }

    return dcGoals;
}

// Calculate 1X2 + Goals combinations
function derive1X2Goals(markets) {
    let result1X2Goals = {};

    STANDARD_LINES.forEach(line => {
        result1X2Goals[line] = {
            'Home': { over: 0, under: 0 },
            'Draw': { over: 0, under: 0 },
            'Away': { over: 0, under: 0 }
        };
    });

    const matrix = appState.jointFull.matrix;
    const size = matrix.length;

    for (let h = 0; h < size; h++) {
        for (let a = 0; a < size; a++) {
            const p = matrix[h][a];
            const total = h + a;
            const isHome = h > a;
            const isDraw = h === a;
            const isAway = h < a;

            STANDARD_LINES.forEach(line => {
                const isOver = total > line;
                const bucket = isOver ? 'over' : 'under';

                if (isHome) result1X2Goals[line]['Home'][bucket] += p;
                else if (isDraw) result1X2Goals[line]['Draw'][bucket] += p;
                else result1X2Goals[line]['Away'][bucket] += p;
            });
        }
    }

    return result1X2Goals;
}

function applyMargin(probs, marginPct) {
    const marginDecimal = marginPct / 100;
    const targetSum = 1 + marginDecimal;
    const currentSum = probs.reduce((a, b) => a + b, 0);
    return probs.map(p => (currentSum === 0 ? 0 : (p / currentSum) * targetSum));
}

const getOdds = (prob) => (prob <= 0.0001 ? 0 : 1 / prob);
const formatProb = (p) => (p * 100).toFixed(2) + '%';
const formatOddsVal = (o) => o > 0 ? o.toFixed(2) : '-';

// --- Rendering ---

function renderAllMarkets() {
    const { marginFull, marginH1, marginH2 } = appState.inputs;

    // Full Time
    const marketsFull = deriveMarkets(appState.jointFull);
    renderListMarkets(marketsFull, 'summary', marginFull, true);
    renderListMarkets(marketsFull, 'tab-full', marginFull, true);
    renderGrid('cs-grid-full', appState.jointFull, 'full', 'warning-full');
    updateSelectionPanel('full', marketsFull.map, marginFull);

    // Calculate new combination markets
    appState.htftData = deriveHalfTimeFullTime(appState.jointH1, appState.jointH2);
    appState.scoresByGoals = deriveFinalScoreByGoals(appState.jointFull);
    appState.dcGoals = deriveDoubleChanceGoals(marketsFull);
    appState.result1X2Goals = derive1X2Goals(marketsFull);

    // H1
    const marketsH1 = deriveMarkets(appState.jointH1);
    renderListMarkets(marketsH1, 'tab-h1', marginH1, false);
    renderGrid('cs-grid-h1', appState.jointH1, 'h1', 'warning-h1');
    updateSelectionPanel('h1', marketsH1.map, marginH1);

    // H2
    const marketsH2 = deriveMarkets(appState.jointH2);
    renderListMarkets(marketsH2, 'tab-h2', marginH2, false);
    renderGrid('cs-grid-h2', appState.jointH2, 'h2', 'warning-h2');
    updateSelectionPanel('h2', marketsH2.map, marginH2);
}

function renderListMarkets(markets, context, margin, isFullTime) {
    const row = (lbl, p, ap) => `
        <div class="market-item">
            <span>${lbl}</span>
            <div class="market-vals">
                <span class="prob-val">${formatProb(p)}</span>
                <span class="odds-val">${formatOddsVal(getOdds(ap))}</span>
            </div>
        </div>`;

    // 1X2
    const p1x2 = [markets.homeWin, markets.draw, markets.awayWin];
    const a1x2 = applyMargin(p1x2, margin);

    if (context.includes('summary')) {
        const el = document.getElementById('summary-1x2');
        if(el) el.innerHTML = row(appState.inputs.homeTeam, p1x2[0], a1x2[0]) +
                              row('Draw', p1x2[1], a1x2[1]) +
                              row(appState.inputs.awayTeam, p1x2[2], a1x2[2]);

        // Summary OU
        const pO = markets.overs[2.5] || 0;
        const aOU = applyMargin([pO, 1-pO], margin);
        document.getElementById('summary-ou').innerHTML = row('Over 2.5', pO, aOU[0]) + row('Under 2.5', 1-pO, aOU[1]);

        // Summary BTTS
        const pB = [markets.bttsYes, markets.bttsNo];
        const aB = applyMargin(pB, margin);
        document.getElementById('summary-btts').innerHTML = row('Yes', pB[0], aB[0]) + row('No', pB[1], aB[1]);
        return;
    }

    if (isFullTime) {
        // For Full Time, render into card-based layout
        renderFullTimeMarketsToCards(markets, margin, row);
        return;
    }

    // For H1 and H2, render into simple card layout
    const suffix = context.split('-')[1];
    const container = document.getElementById(`markets-${suffix}`);
    if (!container) return;

    let html = '';

    // 1X2 Section
    html += row('Home Win', p1x2[0], a1x2[0]);
    html += row('Draw', p1x2[1], a1x2[1]);
    html += row('Away Win', p1x2[2], a1x2[2]);

    // BTTS
    const pB = [markets.bttsYes, markets.bttsNo];
    const aB = applyMargin(pB, margin);
    html += row('BTTS Yes', pB[0], aB[0]);
    html += row('BTTS No', pB[1], aB[1]);

    // Goals
    STANDARD_LINES.forEach(line => {
        const pO = markets.overs[line] || 0;
        const aO = applyMargin([pO, 1-pO], margin);
        html += row(`Over ${line}`, pO, aO[0]);
        html += row(`Under ${line}`, 1-pO, aO[1]);
    });

    container.innerHTML = html;
}

// Render Full Time markets into card-based layout
function renderFullTimeMarketsToCards(markets, margin, row) {
    const p1x2 = [markets.homeWin, markets.draw, markets.awayWin];
    const a1x2 = applyMargin(p1x2, margin);
    const pB = [markets.bttsYes, markets.bttsNo];
    const aB = applyMargin(pB, margin);

    // Basic Markets Card
    let basicHtml = '';
    basicHtml += row('Home Win', p1x2[0], a1x2[0]);
    basicHtml += row('Draw', p1x2[1], a1x2[1]);
    basicHtml += row('Away Win', p1x2[2], a1x2[2]);
    basicHtml += row('BTTS Yes', pB[0], aB[0]);
    basicHtml += row('BTTS No', pB[1], aB[1]);

    const pWTN = [markets.winToNilHome, 1-markets.winToNilHome];
    const aWTN = applyMargin(pWTN, margin);
    basicHtml += row('Win to Nil (Home)', pWTN[0], aWTN[0]);

    STANDARD_LINES.forEach(line => {
        const pO = markets.overs[line] || 0;
        const aO = applyMargin([pO, 1-pO], margin);
        basicHtml += row(`Over ${line}`, pO, aO[0]);
        basicHtml += row(`Under ${line}`, 1-pO, aO[1]);
    });
    document.getElementById('basic-markets-full').innerHTML = basicHtml;

    // Halftime/Full Time Card
    if (appState.htftData) {
        let htftHtml = '';
        const htftOrder = ['H-H', 'H-D', 'H-A', 'D-H', 'D-D', 'D-A', 'A-H', 'A-D', 'A-A'];
        const htftLabels = {
            'H-H': 'Home/Home', 'H-D': 'Home/Draw', 'H-A': 'Home/Away',
            'D-H': 'Draw/Home', 'D-D': 'Draw/Draw', 'D-A': 'Draw/Away',
            'A-H': 'Away/Home', 'A-D': 'Away/Draw', 'A-A': 'Away/Away'
        };
        const htftProbs = htftOrder.map(k => appState.htftData.htft[k]);
        const htftAdjusted = applyMargin(htftProbs, margin);
        htftOrder.forEach((k, i) => {
            htftHtml += row(htftLabels[k], htftProbs[i], htftAdjusted[i]);
        });
        document.getElementById('htft-markets-full').innerHTML = htftHtml;
    }

    // Double Chance + Goals Card
    if (appState.dcGoals) {
        let dcHtml = '';
        const dcLabels = { '1X': 'Home or Draw', 'X2': 'Draw or Away', '12': 'Home or Away' };
        [2.5].forEach(line => {
            const dcData = appState.dcGoals[line];
            ['1X', 'X2', '12'].forEach(dc => {
                const pOver = dcData[dc].over;
                const pUnder = dcData[dc].under;
                const adjusted = applyMargin([pOver, pUnder], margin);
                dcHtml += row(`${dcLabels[dc]} & Over ${line}`, pOver, adjusted[0]);
                dcHtml += row(`${dcLabels[dc]} & Under ${line}`, pUnder, adjusted[1]);
            });
        });
        document.getElementById('dc-goals-full').innerHTML = dcHtml;
    }

    // 1X2 + Goals Card
    if (appState.result1X2Goals) {
        let resultGoalsHtml = '';
        [2.5].forEach(line => {
            const data = appState.result1X2Goals[line];
            ['Home', 'Draw', 'Away'].forEach(result => {
                const pOver = data[result].over;
                const pUnder = data[result].under;
                const adjusted = applyMargin([pOver, pUnder], margin);
                resultGoalsHtml += row(`${result} & Over ${line}`, pOver, adjusted[0]);
                resultGoalsHtml += row(`${result} & Under ${line}`, pUnder, adjusted[1]);
            });
        });
        document.getElementById('result-goals-full').innerHTML = resultGoalsHtml;
    }

    // Exact Goals Table
    const goalsTable = document.getElementById('goals-full');
    let gHtml = `<thead><tr><th>Total</th><th>Prob</th><th>Odds</th></tr></thead><tbody>`;
    for(let t=0; t<=MAX_GOALS; t++) {
        const p = markets.exactTotals[t];
        const odd = getOdds(p * (1 + margin/100));
        gHtml += `<tr><td>${t}</td><td>${formatProb(p)}</td><td>${formatOddsVal(odd)}</td></tr>`;
    }
    goalsTable.innerHTML = gHtml + `</tbody>`;

    // Render Scores by Goals breakdown
    if (appState.scoresByGoals) {
        renderScoresByGoals(appState.scoresByGoals, margin);
    }
}

// Render Final Score by Goals breakdown
function renderScoresByGoals(scoresByGoals, margin) {
    const container = document.getElementById('scores-by-goals-full');
    if (!container) return;

    let html = '';
    for (let total = 0; total <= MAX_GOALS; total++) {
        const scores = scoresByGoals[total];
        if (!scores || scores.length === 0) continue;

        // Filter to show only scores with meaningful probability
        const significantScores = scores.filter(s => s.prob > 0.001);
        if (significantScores.length === 0) continue;

        html += `<div class="goal-group">
            <h6>${total} Goal${total !== 1 ? 's' : ''}</h6>
            <div class="score-list">`;

        significantScores.slice(0, 10).forEach(item => {
            const adjustedProb = item.prob * (1 + margin/100);
            const odds = getOdds(adjustedProb);
            html += `<div class="score-item">
                <span class="score-label">${item.score}</span>
                <span class="score-prob">${formatProb(item.prob)}</span>
                <span class="score-odds">${formatOddsVal(odds)}</span>
            </div>`;
        });

        html += `</div></div>`;
    }

    container.innerHTML = html;
}

// --- Grid & Selection Logic ---

function renderGrid(tableId, jointDist, period, warningId) {
    const table = document.getElementById(tableId);
    const matrix = jointDist.matrix;
    const warningEl = document.getElementById(warningId);
    
    warningEl.textContent = jointDist.tailProb > 0.01 
        ? `Tail probability > 7 goals: ${(jointDist.tailProb * 100).toFixed(2)}%` 
        : '';

    table.innerHTML = '';
    let thead = '<thead><tr><th>H \\ A</th>';
    for (let j = 0; j <= MAX_GOALS; j++) thead += `<th>${j}</th>`;
    thead += '</tr></thead>';
    table.innerHTML = thead;

    let tbody = document.createElement('tbody');
    for (let i = 0; i <= MAX_GOALS; i++) {
        let tr = document.createElement('tr');
        let th = document.createElement('th');
        th.innerText = i;
        tr.appendChild(th);

        for (let j = 0; j <= MAX_GOALS; j++) {
            let td = document.createElement('td');
            let prob = matrix[i][j];
            let key = `${i}-${j}`;
            
            td.innerText = (prob * 100).toFixed(2);
            
            // Selection State
            if (selections[period].has(key)) td.classList.add('selected');

            // Click Handler
            td.onclick = () => {
                if (selections[period].has(key)) {
                    selections[period].delete(key);
                    td.classList.remove('selected');
                } else {
                    selections[period].add(key);
                    td.classList.add('selected');
                }
                // Need to re-derive map or pass it? Better to store calculated maps in appState or re-use logic
                // For simplicity, re-run full render or just selection update?
                // Re-running deriveMarkets is cheap enough.
                const dist = period === 'full' ? appState.jointFull : period === 'h1' ? appState.jointH1 : appState.jointH2;
                const m = deriveMarkets(dist);
                const margin = period === 'full' ? appState.inputs.marginFull : period === 'h1' ? appState.inputs.marginH1 : appState.inputs.marginH2;
                updateSelectionPanel(period, m.map, margin);
            };
            tr.appendChild(td);
        }
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
}

function updateSelectionPanel(period, probabilityMap, margin) {
    const panelId = `selection-panel-${period}`;
    const panel = document.getElementById(panelId);
    const selSet = selections[period];
    
    if (selSet.size === 0) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');

    let totalProb = 0;
    let descList = [];

    // Sort selections for display
    const sortedKeys = Array.from(selSet).sort((a,b) => {
        const [h1, a1] = a.split('-').map(Number);
        const [h2, a2] = b.split('-').map(Number);
        return h1 - h2 || a1 - a2;
    });

    sortedKeys.forEach(key => {
        totalProb += (probabilityMap[key] || 0);
        descList.push(key.replace('-', ':'));
    });

    // Stats
    const fairOdds = getOdds(totalProb);
    // Market odds for a custom aggregation:
    // Implied Prob = True Prob * (1 + Margin)
    // Odds = 1 / Implied Prob
    const impliedProb = totalProb * (1 + margin/100);
    const marketOdds = getOdds(impliedProb);

    // Update DOM
    document.getElementById(`sel-count-${period}`).innerText = selSet.size;
    document.getElementById(`sel-prob-${period}`).innerText = formatProb(totalProb);
    document.getElementById(`sel-fair-${period}`).innerText = formatOddsVal(fairOdds);
    document.getElementById(`sel-market-${period}`).innerText = formatOddsVal(marketOdds);
    
    // List preview
    const listEl = document.getElementById(`sel-list-${period}`);
    listEl.innerText = descList.join(', ');
}

// Global scope for HTML onclick
window.clearSelection = function(period) {
    selections[period].clear();
    const table = document.getElementById(`cs-grid-${period}`);
    if(table) {
        table.querySelectorAll('.selected').forEach(td => td.classList.remove('selected'));
    }
    const panel = document.getElementById(`selection-panel-${period}`);
    if(panel) panel.classList.add('hidden');
};
