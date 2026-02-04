/**
 * Groww TradeBot Dashboard
 * Client-side interactivity for navigation, theming, charts, and mock data.
 */

(function () {
    'use strict';

    // ========================================
    // Navigation
    // ========================================

    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');
    const pageTitle = document.querySelector('.page-title');

    const pageTitles = {
        dashboard: 'Dashboard',
        trades: 'Trade History',
        'rl-agent': 'RL Agent',
        indicators: 'Technical Indicators',
        'paper-trading': 'Paper Trading',
        settings: 'Settings',
    };

    function navigateTo(pageId) {
        navItems.forEach(item => item.classList.remove('active'));
        pages.forEach(page => page.classList.remove('active'));

        const targetNav = document.querySelector(`.nav-item[data-page="${pageId}"]`);
        const targetPage = document.getElementById(`page-${pageId}`);

        if (targetNav) targetNav.classList.add('active');
        if (targetPage) targetPage.classList.add('active');
        if (pageTitle) pageTitle.textContent = pageTitles[pageId] || 'Dashboard';
    }

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navigateTo(item.dataset.page);
        });
    });

    // Handle "View all" links
    document.querySelectorAll('[data-page]').forEach(el => {
        if (el.tagName === 'A') {
            el.addEventListener('click', (e) => {
                e.preventDefault();
                navigateTo(el.dataset.page);
            });
        }
    });

    // ========================================
    // Theme Toggle
    // ========================================

    const themeToggle = document.getElementById('themeToggle');
    const lightBtn = document.getElementById('lightThemeBtn');
    const darkBtn = document.getElementById('darkThemeBtn');

    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('groww-theme', theme);

        if (lightBtn && darkBtn) {
            lightBtn.classList.toggle('btn-primary', theme === 'light');
            lightBtn.classList.toggle('btn-ghost', theme !== 'light');
            darkBtn.classList.toggle('btn-primary', theme === 'dark');
            darkBtn.classList.toggle('btn-ghost', theme !== 'dark');
        }
    }

    // Load saved theme
    const savedTheme = localStorage.getItem('groww-theme') || 'dark';
    setTheme(savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            setTheme(current === 'dark' ? 'light' : 'dark');
        });
    }

    if (lightBtn) lightBtn.addEventListener('click', () => setTheme('light'));
    if (darkBtn) darkBtn.addEventListener('click', () => setTheme('dark'));

    // ========================================
    // Bot Toggle
    // ========================================

    const toggleBtn = document.getElementById('toggleBot');
    const statusDot = document.querySelector('.status-dot');
    const statusLabel = document.querySelector('.status-label');
    let botRunning = false;

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            botRunning = !botRunning;
            statusDot.className = `status-dot ${botRunning ? 'status-active' : 'status-inactive'}`;
            statusLabel.textContent = botRunning ? 'Bot Running' : 'Bot Offline';
            toggleBtn.textContent = botRunning ? 'Stop Bot' : 'Start Bot';
            toggleBtn.classList.toggle('btn-primary', !botRunning);
            toggleBtn.classList.toggle('btn-danger', botRunning);
        });
    }

    // ========================================
    // Market Status
    // ========================================

    function updateMarketStatus() {
        const el = document.getElementById('marketStatus');
        if (!el) return;

        const now = new Date();
        const hours = now.getHours();
        const minutes = now.getMinutes();
        const day = now.getDay();
        const timeNum = hours * 100 + minutes;

        // Indian market: Mon-Fri, 9:15 AM to 3:30 PM IST
        const isWeekday = day >= 1 && day <= 5;
        const isMarketHours = timeNum >= 915 && timeNum <= 1530;

        if (isWeekday && isMarketHours) {
            el.textContent = 'Open';
            el.style.color = 'var(--color-success)';
        } else if (isWeekday && timeNum >= 900 && timeNum < 915) {
            el.textContent = 'Pre-Open';
            el.style.color = 'var(--color-warning)';
        } else {
            el.textContent = 'Closed';
            el.style.color = 'var(--text-tertiary)';
        }
    }

    updateMarketStatus();
    setInterval(updateMarketStatus, 60000);

    // ========================================
    // P&L Chart (Canvas)
    // ========================================

    function drawPnLChart() {
        const canvas = document.getElementById('pnlCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * 2;
        canvas.height = rect.height * 2;
        ctx.scale(2, 2);

        const w = rect.width;
        const h = rect.height;
        const padding = { top: 20, right: 20, bottom: 30, left: 60 };

        // Mock P&L data points
        const data = [
            0, 1200, 800, 2400, 1800, 3200, 2600, 4100, 3500, 5200,
            4800, 6100, 5400, 7200, 6800, 8400, 7600, 9100, 10200,
            9800, 11400, 10800, 12450
        ];

        const maxVal = Math.max(...data) * 1.1;
        const minVal = Math.min(...data) - Math.abs(Math.min(...data)) * 0.1;
        const range = maxVal - minVal;

        const chartW = w - padding.left - padding.right;
        const chartH = h - padding.top - padding.bottom;

        // Background
        ctx.clearRect(0, 0, w, h);

        // Grid lines
        const style = getComputedStyle(document.documentElement);
        const gridColor = style.getPropertyValue('--border-color').trim();
        const textColor = style.getPropertyValue('--text-tertiary').trim();
        const successColor = style.getPropertyValue('--color-success').trim();

        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 0.5;

        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (chartH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(w - padding.right, y);
            ctx.stroke();

            const val = maxVal - (range / 4) * i;
            ctx.fillStyle = textColor;
            ctx.font = '11px Inter, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`${Math.round(val).toLocaleString('en-IN')}`, padding.left - 8, y + 4);
        }

        // Draw area fill
        const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
        gradient.addColorStop(0, 'rgba(0, 200, 83, 0.2)');
        gradient.addColorStop(1, 'rgba(0, 200, 83, 0)');

        ctx.beginPath();
        ctx.moveTo(padding.left, h - padding.bottom);

        data.forEach((val, i) => {
            const x = padding.left + (chartW / (data.length - 1)) * i;
            const y = padding.top + chartH - ((val - minVal) / range) * chartH;
            ctx.lineTo(x, y);
        });

        ctx.lineTo(padding.left + chartW, h - padding.bottom);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw line
        ctx.beginPath();
        ctx.strokeStyle = successColor;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';

        data.forEach((val, i) => {
            const x = padding.left + (chartW / (data.length - 1)) * i;
            const y = padding.top + chartH - ((val - minVal) / range) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.stroke();

        // End dot
        const lastX = padding.left + chartW;
        const lastY = padding.top + chartH - ((data[data.length - 1] - minVal) / range) * chartH;
        ctx.beginPath();
        ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
        ctx.fillStyle = successColor;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(lastX, lastY, 7, 0, Math.PI * 2);
        ctx.strokeStyle = successColor;
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    drawPnLChart();
    window.addEventListener('resize', drawPnLChart);

    // ========================================
    // Mock Trade Data
    // ========================================

    const mockTrades = [
        { date: '2026-02-04', time: '09:45:12', symbol: 'NIFTY 24500 CE', type: 'CE', entry: 245.50, exit: 278.30, qty: 50, pnl: 1640, rlScore: 0.82, status: 'Closed' },
        { date: '2026-02-04', time: '10:12:30', symbol: 'NIFTY 24400 PE', type: 'PE', entry: 180.00, exit: null, qty: 25, pnl: 875, rlScore: 0.71, status: 'Open' },
        { date: '2026-02-04', time: '10:30:45', symbol: 'BANKNIFTY 52000 CE', type: 'CE', entry: 410.25, exit: 385.00, qty: 25, pnl: -631, rlScore: 0.58, status: 'SL Hit' },
        { date: '2026-02-03', time: '09:30:15', symbol: 'NIFTY 24600 CE', type: 'CE', entry: 190.00, exit: 225.50, qty: 50, pnl: 1775, rlScore: 0.89, status: 'Closed' },
        { date: '2026-02-03', time: '11:05:20', symbol: 'NIFTY 24300 PE', type: 'PE', entry: 210.75, exit: 195.00, qty: 25, pnl: -394, rlScore: 0.45, status: 'SL Hit' },
        { date: '2026-02-03', time: '13:15:10', symbol: 'BANKNIFTY 51800 CE', type: 'CE', entry: 320.00, exit: 380.50, qty: 50, pnl: 3025, rlScore: 0.91, status: 'Closed' },
    ];

    function populateTradeTable() {
        const tbody = document.getElementById('tradeTableBody');
        if (!tbody) return;

        tbody.innerHTML = mockTrades.map(t => {
            const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
            const pnlStr = t.pnl >= 0 ? `+\u20B9${t.pnl.toLocaleString('en-IN')}` : `-\u20B9${Math.abs(t.pnl).toLocaleString('en-IN')}`;
            const typeBadge = t.type === 'CE' ? 'badge-success' : 'badge-danger';
            const statusBadge = t.status === 'Closed' ? 'badge-success' : t.status === 'Open' ? 'badge-warning' : 'badge-danger';

            return `<tr>
                <td>${t.date}</td>
                <td>${t.time}</td>
                <td class="font-mono">${t.symbol}</td>
                <td><span class="badge ${typeBadge}">${t.type}</span></td>
                <td>${t.entry.toFixed(2)}</td>
                <td>${t.exit ? t.exit.toFixed(2) : '--'}</td>
                <td>${t.qty}</td>
                <td class="${pnlClass}">${pnlStr}</td>
                <td><code>${t.rlScore.toFixed(2)}</code></td>
                <td><span class="badge ${statusBadge}">${t.status}</span></td>
            </tr>`;
        }).join('');
    }

    populateTradeTable();

})();
