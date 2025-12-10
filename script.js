/**
 * ZIP Football Calculator
 * * Implements Zero-Inflated Poisson distribution for football score modelling.
 * Features:
 * - Conversion between xG inputs and Supremacy/Expectancy
 * - Zero-Inflation parameter handling
 * - Market derivation (1X2, O/U, BTTS, Correct Score)
 * - Margin application
 */

// --- Constants & Config ---
const MAX_GOALS = 7; // Grid size 0-7
const STANDARD_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];

// Store calculation results globally to allow margin updates without re-computing distribution
let appState = {
    jointFull: null,
    jointH1: null,
    jointH2: null,
    inputs: null
};

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    // Perform initial calculation on load
    calculateAndRender();
});

// --- Event Listeners ---
function initEventListeners() {
    // Input Mode Toggle
    const modeRadios = document.getElementsByName('inputMode');
    modeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            toggleInputMode(e.target.value);
        });
    });

    // ZIP Toggle
    document.getElementById('useZip').addEventListener('change', (e) => {
        toggleZipInputs(e.target.checked);
    });

    // Calculate Button
    document.getElementById('calcBtn').addEventListener('click', calculateAndRender);

    // Margin Change (Re-render odds only)
    document.getElementById('marginPct').addEventListener('input', () => {
        if(appState.jointFull) {
            renderAllMarkets();
        }
    });

    // Tabs
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            // Remove active class from all
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add to click target
            e.target.classList.add('active');
            const contentId = `tab-${e.target.dataset.tab}`;
            document.getElementById(contentId).classList.add('active');
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
    
    // Base object
    let data = {
        homeTeam: document.getElementById('homeTeam').value || 'Home',
        awayTeam: document.getElementById('awayTeam').value || 'Away',
        lambdaHomeFull: 0,
        lambdaAwayFull: 0,
        piHome: 0,
        piAway: 0,
        halfFactor: parseFloat(document.getElementById('halfFactor').value),
        marginPct: parseFloat(document.getElementById('marginPct').value),
        isValid: true,
        errorMsg: ''
    };

    // 1.3 Validate & Convert Inputs
    if (mode === 'xg') {
        data.lambdaHomeFull = parseFloat(document.getElementById('xgHome').value);
        data.lambdaAwayFull = parseFloat(document.getElementById('xgAway').value);
        
        if (data.lambdaHomeFull < 0 || data.lambdaAwayFull < 0) {
            data.isValid = false;
            data.errorMsg = "xG values must be non-negative.";
        }
    } else {
        // Supremacy / Expectancy conversion
        // Sup = H - A
        // Exp = H + A
        // H = (Exp + Sup) / 2
        // A = (Exp - Sup) / 2
        const sup = parseFloat(document.getElementById('supremacy').value);
        const exp = parseFloat(document.getElementById('expectancy').value);
        
        if (exp <= 0) {
            data.isValid = false;
            data.errorMsg = "Total expectancy must be greater than 0.";
        } else {
            data.lambdaHomeFull = (exp + sup) / 2;
            data.lambdaAwayFull = (exp - sup) / 2;
            
            if (data.lambdaHomeFull < 0 || data.lambdaAwayFull < 0) {
                data.isValid = false;
                data.errorMsg = "Implied team goals are negative. Check Supremacy vs Expectancy.";
            }
        }
    }

    // ZIP params
    if (useZip) {
        data.piHome = parseFloat(document.getElementById('piHome').value);
        data.piAway = parseFloat(document.getElementById('piAway').value);
        
        // Clamp 0-1
        if (data.piHome < 0 || data.piHome > 1 || data.piAway < 0 || data.piAway > 1) {
            data.isValid = false;
            data.errorMsg = "Pi (Zero-Inflation) must be between 0 and 1.";
        }
    }

    // Margin validation
    if (data.marginPct < 0) data.marginPct = 0;

    return data;
}

// --- Mathematical Model ---

const factorialCache = [1, 1, 2, 6, 24, 120, 720, 5040]; // 0! to 7!
function factorial(n) {
    if (n < 0) return 1;
    if (n < factorialCache.length) return factorialCache[n];
    let res = factorialCache[factorialCache.length - 1];
    for (let i = factorialCache.length; i <= n; i++) res *= i;
    return res;
}

/**
 * 1.2 Zero-Inflated Poisson PMF
 * Returns array of size maxGoals + 1 containing probabilities.
 * Also returns sum to calculate tail probability.
 */
function computePoissonPMF(lambda, maxGoals, pi) {
    let probs = [];
    let sum = 0;
    const expLambda = Math.exp(-lambda);

    for (let k = 0; k <= maxGoals; k++) {
        let pStandard = (expLambda * Math.pow(lambda, k)) / factorial(k);
        let pZip = 0;

        if (k === 0) {
            pZip = pi + (1 - pi) * pStandard;
        } else {
            pZip = (1 - pi) * pStandard;
        }
        
        probs.push(pZip);
        sum += pZip;
    }
    
    return { probs, tail: 1 - sum };
}

/**
 * 1.5 Compute Joint Distribution Matrix
 * Returns 2D array [home][away] and tail stats
 */
function computeJointDistribution(lambdaH, lambdaA, piH, piA, maxGoals) {
    const distH = computePoissonPMF(lambdaH, maxGoals, piH);
    const distA = computePoissonPMF(lambdaA, maxGoals, piA);

    let matrix = [];
    let totalProb = 0;

    for (let i = 0; i <= maxGoals; i++) {
        matrix[i] = [];
        for (let j = 0; j <= maxGoals; j++) {
            // Independent events: P(H) * P(A)
            let jointP = distH.probs[i] * distA.probs[j];
            matrix[i][j] = jointP;
            totalProb += jointP;
        }
    }

    return {
        matrix: matrix,
        tailProb: 1 - totalProb,
        distH: distH,
        distA: distA
    };
}

// --- Market Derivation ---

/**
 * 2.1 - 2.5 Calculate Market Probabilities
 */
function deriveMarkets(jointDist) {
    const matrix = jointDist.matrix;
    const size = matrix.length; // 8 (0-7)
    
    let markets = {
        // 1X2
        homeWin: 0,
        draw: 0,
        awayWin: 0,
        
        // BTTS
        bttsYes: 0,
        bttsNo: 0,
        
        // Win to Nil / Clean Sheet
        winToNilHome: 0,
        winToNilAway: 0,
        cleanSheetHome: 0,
        cleanSheetAway: 0,
        
        // Over/Under maps
        overs: {}, 
        
        // Exact Totals
        exactTotals: []
    };

    // Initialize exact totals array (0 to 14 theoretically, but we limit display)
    let maxTotal = (size - 1) * 2;
    for(let t=0; t<=maxTotal; t++) markets.exactTotals[t] = 0;

    // Loop grid
    for (let h = 0; h < size; h++) {
        for (let a = 0; a < size; a++) {
            const p = matrix[h][a];
            const total = h + a;

            // 1X2
            if (h > a) markets.homeWin += p;
            else if (h === a) markets.draw += p;
            else markets.awayWin += p;

            // BTTS
            if (h > 0 && a > 0) markets.bttsYes += p;

            // Win to Nil
            if (h > 0 && a === 0) markets.winToNilHome += p;
            if (a > 0 && h === 0) markets.winToNilAway += p;

            // Clean Sheet (Home CS means Away = 0)
            if (a === 0) markets.cleanSheetHome += p;
            if (h === 0) markets.cleanSheetAway += p;

            // Exact Totals
            if (total <= maxTotal) markets.exactTotals[total] += p;

            // Over/Under Accumulation
            STANDARD_LINES.forEach(line => {
                if (total > line) {
                    markets.overs[line] = (markets.overs[line] || 0) + p;
                }
            });
        }
    }
    
    markets.bttsNo = 1 - markets.bttsYes;

    // Normalize 1X2 sum to exactly 1 for calculation stability (ignoring grid tail)
    // In a strict model, we might treat tail as a distribution, but here we work with the grid.
    // However, usually, we calculate derived markets based on infinite sums (CDF).
    // Given the grid constraint, we use the grid sum.
    const sum1x2 = markets.homeWin + markets.draw + markets.awayWin;
    if (sum1x2 > 0) {
        markets.homeWin /= sum1x2;
        markets.draw /= sum1x2;
        markets.awayWin /= sum1x2;
    }

    return markets;
}

/**
 * 2.1 Margin Application logic
 * Target Sum = 1 + margin_decimal
 * Adjusted Prob = Raw Prob / Sum(Raw Probs) * Target Sum
 */
function applyMargin(probs, marginPct) {
    const marginDecimal = marginPct / 100;
    const targetSum = 1 + marginDecimal;
    const currentSum = probs.reduce((a, b) => a + b, 0);
    
    return probs.map(p => {
        if (currentSum === 0) return 0;
        return (p / currentSum) * targetSum;
    });
}

function getOdds(prob) {
    if (prob <= 0.0001) return 0;
    return 1 / prob;
}

// --- Formatting Helpers ---
const formatProb = (p) => (p * 100).toFixed(2) + '%';
const formatOddsVal = (o) => o > 0 ? o.toFixed(2) : '-';

// --- Rendering Logic ---

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

    // 1. Calculate Distributions
    // Full Time
    appState.jointFull = computeJointDistribution(
        inputs.lambdaHomeFull, 
        inputs.lambdaAwayFull, 
        inputs.piHome, 
        inputs.piAway, 
        MAX_GOALS
    );

    // 1.4 Halves (Scaling)
    const factor = inputs.halfFactor;
    
    // 1st Half
    appState.jointH1 = computeJointDistribution(
        inputs.lambdaHomeFull * factor, 
        inputs.lambdaAwayFull * factor, 
        inputs.piHome, 
        inputs.piAway, 
        MAX_GOALS
    );

    // 2nd Half
    appState.jointH2 = computeJointDistribution(
        inputs.lambdaHomeFull * factor, // Assumed symmetric scaling based on prompt 1.4
        inputs.lambdaAwayFull * factor, 
        inputs.piHome, 
        inputs.piAway, 
        MAX_GOALS
    );

    renderAllMarkets();
}

function renderAllMarkets() {
    const margin = appState.inputs.marginPct;

    // Render Summary & Full Time Tab
    const marketsFull = deriveMarkets(appState.jointFull);
    renderMarkets(marketsFull, 'summary', margin, true); // Update summary cards
    renderMarkets(marketsFull, 'tab-full', margin, true);
    renderGrid('cs-grid-full', appState.jointFull, margin, 'warning-full', 'selected-score-full');

    // Render H1 Tab
    const marketsH1 = deriveMarkets(appState.jointH1);
    renderMarkets(marketsH1, 'tab-h1', margin, false);
    renderGrid('cs-grid-h1', appState.jointH1, margin, 'warning-h1', 'selected-score-h1');

    // Render H2 Tab
    const marketsH2 = deriveMarkets(appState.jointH2);
    renderMarkets(marketsH2, 'tab-h2', margin, false);
    renderGrid('cs-grid-h2', appState.jointH2, margin, 'warning-h2', 'selected-score-h2');
}

/**
 * Generic renderer for list-based markets
 * containerPrefix: 'summary' or 'tab-full', etc.
 */
function renderMarkets(markets, context, margin, isFullTime) {
    
    // Helper to generate HTML row
    const row = (label, prob, adjProb) => `
        <div class="market-item">
            <span>${label}</span>
            <div class="market-vals">
                <span class="prob-val">${formatProb(prob)}</span>
                <span class="odds-val">${formatOddsVal(getOdds(adjProb))}</span>
            </div>
        </div>
    `;

    // 1. 1X2 (Only for Summary and Full Time tab)
    if (isFullTime || context.includes('summary')) {
        const probs1x2 = [markets.homeWin, markets.draw, markets.awayWin];
        const adj1x2 = applyMargin(probs1x2, margin);
        
        let html1x2 = row(appState.inputs.homeTeam, probs1x2[0], adj1x2[0]);
        html1x2 += row('Draw', probs1x2[1], adj1x2[1]);
        html1x2 += row(appState.inputs.awayTeam, probs1x2[2], adj1x2[2]);

        // Place in Summary
        if (isFullTime) {
            const sumEl = document.getElementById('summary-1x2');
            if(sumEl) sumEl.innerHTML = html1x2;
        }
        
        // Place in Tab Side Panel (if exists)
        const tabEl = document.getElementById(`markets-${context.split('-')[1]}`);
        if (tabEl && isFullTime) {
            tabEl.innerHTML = `<h6>Match Result</h6>${html1x2}`;
        }
    }

    // 2. Over/Under 2.5 (Summary)
    if (context.includes('summary')) {
        const pOver = markets.overs[2.5] || 0;
        const pUnder = 1 - pOver;
        const adjOU = applyMargin([pOver, pUnder], margin);
        
        let htmlOU = row('Over 2.5', pOver, adjOU[0]);
        htmlOU += row('Under 2.5', pUnder, adjOU[1]);
        
        const el = document.getElementById('summary-ou');
        if(el) el.innerHTML = htmlOU;
    }

    // 3. BTTS (Summary + Tab)
    const probsBtts = [markets.bttsYes, markets.bttsNo];
    const adjBtts = applyMargin(probsBtts, margin);
    const htmlBtts = row('Yes', probsBtts[0], adjBtts[0]) + row('No', probsBtts[1], adjBtts[1]);

    if (context.includes('summary')) {
        const el = document.getElementById('summary-btts');
        if(el) el.innerHTML = htmlBtts;
    }

    // Tab Specific Content
    if (!context.includes('summary')) {
        const suffix = context.split('-')[1]; // full, h1, h2
        const container = document.getElementById(`markets-${suffix}`);
        if (!container) return;

        let content = '';

        // Add 1X2 for halves if needed (Using derived markets)
        if (!isFullTime) {
            const probs1x2 = [markets.homeWin, markets.draw, markets.awayWin];
            const adj1x2 = applyMargin(probs1x2, margin);
            content += `<h6>1X2 (Period)</h6>`;
            content += row('Home', probs1x2[0], adj1x2[0]);
            content += row('Draw', probs1x2[1], adj1x2[1]);
            content += row('Away', probs1x2[2], adj1x2[2]);
        }

        // BTTS
        content += `<h6>Both Teams To Score</h6>${htmlBtts}`;

        // Over/Under Lines
        content += `<h6>Goal Lines</h6>`;
        STANDARD_LINES.forEach(line => {
            const pOver = markets.overs[line] || 0;
            const pUnder = 1 - pOver;
            const adj = applyMargin([pOver, pUnder], margin);
            content += row(`Over ${line}`, pOver, adj[0]);
            content += row(`Under ${line}`, pUnder, adj[1]);
        });

        // Win to Nil / Clean Sheet (Full Time primarily, but logic works for halves too)
        content += `<h6>Win to Nil / Clean Sheet</h6>`;
        const pWTN = [markets.winToNilHome, 1-markets.winToNilHome];
        const adjWTN = applyMargin(pWTN, margin);
        content += row('Home Win to Nil', pWTN[0], adjWTN[0]);

        const pCS = [markets.cleanSheetHome, 1-markets.cleanSheetHome];
        const adjCS = applyMargin(pCS, margin);
        content += row('Home Clean Sheet', pCS[0], adjCS[0]);

        container.innerHTML += content; // Append to existing (1X2 might be there for full time)

        // Exact Goals Table (Full Time specific visualization)
        if (isFullTime) {
            const goalsTable = document.getElementById('goals-full');
            let goalsHtml = `<thead><tr><th>Total</th><th>Prob</th><th>Odds</th></tr></thead><tbody>`;
            for(let t=0; t<=MAX_GOALS; t++) { // Show up to 7
                const p = markets.exactTotals[t];
                // Margin applied against the set 0-MAX_GOALS isn't strictly fair as set isn't exhaustive, 
                // but we apply margin individually P vs 1-P for UI simplicity or group normalization.
                // Here we calculate odds as 1/P * margin for single outcome approximation
                const odds = 1 / (p / (1 + margin/100)); // Simple inflation
                goalsHtml += `<tr><td>${t}</td><td>${formatProb(p)}</td><td>${formatOddsVal(odds)}</td></tr>`;
            }
            goalsHtml += `</tbody>`;
            goalsTable.innerHTML = goalsHtml;
        }
    }
}

function renderGrid(tableId, jointDist, margin, warningId, selectionPanelId) {
    const table = document.getElementById(tableId);
    const matrix = jointDist.matrix;
    const warningEl = document.getElementById(warningId);
    
    // Tail Warning
    if (jointDist.tailProb > 0.01) {
        warningEl.textContent = `Warning: Tail probability > 7 goals is ${(jointDist.tailProb * 100).toFixed(2)}%. Model may understate high scores.`;
    } else {
        warningEl.textContent = '';
    }

    table.innerHTML = '';
    
    // Header
    let thead = '<thead><tr><th>H \\ A</th>';
    for (let j = 0; j <= MAX_GOALS; j++) thead += `<th>${j}</th>`;
    thead += '</tr></thead>';
    table.innerHTML = thead;

    let tbody = document.createElement('tbody');

    for (let i = 0; i <= MAX_GOALS; i++) {
        let row = document.createElement('tr');
        let th = document.createElement('th');
        th.innerText = i;
        row.appendChild(th);

        for (let j = 0; j <= MAX_GOALS; j++) {
            let td = document.createElement('td');
            let prob = matrix[i][j];
            
            td.innerText = (prob * 100).toFixed(2);
            
            // Interaction
            td.onclick = function() {
                // Remove existing selection in this table
                const prev = table.querySelectorAll('.selected');
                prev.forEach(el => el.classList.remove('selected'));
                
                // Add to current
                td.classList.add('selected');
                
                // Update Panel
                updateSelectionPanel(selectionPanelId, i, j, prob, margin);
            };

            row.appendChild(td);
        }
        tbody.appendChild(row);
    }
    table.appendChild(tbody);
}

function updateSelectionPanel(panelId, homeGoals, awayGoals, prob, margin) {
    const panel = document.getElementById(panelId);
    panel.classList.remove('hidden');
    
    // Apply margin to this single outcome vs The Field logic?
    // Or scaling the whole grid? 
    // Usually Correct Score markets sum to ~1.20 (high margin).
    // Here we use the global margin setting applied to this specific probability 
    // assuming it's part of the complete set.
    // Odds = 1 / (Prob / SumProbs * (1+Margin))
    // Since Prob is from a sum~1 distribution:
    const adjProb = prob / (1 + margin/100); 
    // Wait, prompt formula: p' = p / sum * (1 + margin).
    // Odds = 1 / p'.
    // If we assume sum is 1: p' = p * (1 + margin).
    // Odds = 1 / (p * (1 + margin)).
    // This actually reduces odds (higher price) which is wrong for bookmaking.
    // Bookmaker margin: Implied Prob = True Prob * (1 + Margin).
    // Odds = 1 / Implied Prob.
    // So: Odds = 1 / (p * (1 + margin/100)) is correct for "Odds with Margin".
    
    // WAIT: Standard formula in prompt:
    // Target sum = 1 + margin_decimal.
    // p' = p * (1 + margin_decimal).
    // O' = 1 / p'.
    // Example: P=0.5. Margin 10%. P'=0.55. Odds = 1.81. (Fair was 2.0).
    // This reduces the return, which is correct for a bookmaker.
    
    const impliedProb = prob * (1 + margin/100);
    const marketOdds = getOdds(impliedProb);
    const fairOdds = getOdds(prob);

    panel.querySelector('.sel-score').innerText = `${homeGoals} - ${awayGoals}`;
    panel.querySelector('.sel-prob').innerText = formatProb(prob);
    panel.querySelector('.sel-fair').innerText = formatOddsVal(fairOdds);
    panel.querySelector('.sel-market').innerText = formatOddsVal(marketOdds);
}
