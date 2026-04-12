/**
 * Train pipeline — API + Chart.js (Oiloz palette)
 */
(function () {
    'use strict';

    const cfg = JSON.parse(document.getElementById('train-config').textContent);
    const DEFAULT_FEATURES = cfg.default_features || [];

    let sessionId = null;
    let lastColumns = [];
    let wellNameForSave = '';
    let chartOil = null;
    let chartWater = null;
    let chartChoke = null;

    const COLORS = {
        blue: '#4a90e2',
        blueSoft: 'rgba(74, 144, 226, 0.45)',
        orange: '#ff7a5c',
        orangeSoft: 'rgba(255, 122, 92, 0.45)',
        green: '#10b981',
        slate: '#64748b',
    };

    function toast(msg, isErr) {
        const el = document.getElementById('trainToast');
        el.textContent = msg;
        el.classList.remove('hidden', 'err');
        if (isErr) el.classList.add('err');
        clearTimeout(toast._t);
        toast._t = setTimeout(function () {
            el.classList.add('hidden');
        }, 5000);
    }

    function setBusy(stepId, on) {
        var el = document.getElementById(stepId);
        if (!el) return;
        if (on) el.classList.add('is-busy');
        else el.classList.remove('is-busy');
    }

    function headersJson() {
        var h = { 'Content-Type': 'application/json' };
        if (sessionId) h['X-Session-Id'] = sessionId;
        return h;
    }

    async function postJson(url, body) {
        var r = await fetch(url, {
            method: 'POST',
            headers: headersJson(),
            body: JSON.stringify(Object.assign({ session_id: sessionId }, body || {})),
        });
        var d = await r.json().catch(function () {
            return {};
        });
        if (!r.ok || d.ok === false) {
            throw new Error(d.error || r.statusText || 'Request failed');
        }
        return d;
    }

    async function getJson(url) {
        var sep = url.indexOf('?') >= 0 ? '&' : '?';
        var u = sessionId ? url + sep + 'session_id=' + encodeURIComponent(sessionId) : url;
        var r = await fetch(u, { headers: sessionId ? { 'X-Session-Id': sessionId } : {} });
        var d = await r.json().catch(function () {
            return {};
        });
        if (!r.ok || d.ok === false) {
            throw new Error(d.error || r.statusText || 'Request failed');
        }
        return d;
    }

    function fmtCv(obj) {
        if (!obj) return '—';
        return JSON.stringify(obj, null, 2);
    }

    function destroyCharts() {
        [chartOil, chartWater, chartChoke].forEach(function (c) {
            if (c) {
                c.destroy();
            }
        });
        chartOil = chartWater = chartChoke = null;
    }

    function buildChartsFromRows(rows) {
        if (!rows || !rows.length) return;
        destroyCharts();
        var labels = rows.map(function (r) {
            return r.DATEPRD ? String(r.DATEPRD).slice(0, 10) : '';
        });
        var chokeScale = function (v) {
            var x = Number(v);
            if (isNaN(x)) return 0;
            return x <= 1.25 ? x * 100 : x;
        };
        var oilCtx = document.getElementById('chartOil');
        var waterCtx = document.getElementById('chartWater');
        var chokeCtx = document.getElementById('chartChoke');
        chartOil = new Chart(oilCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Oil aktual',
                        data: rows.map(function (r) {
                            return r.Oil_Actual;
                        }),
                        borderColor: COLORS.blue,
                        tension: 0.2,
                    },
                    // {
                    //     label: 'Oil pred (choke aktual)',
                    //     data: rows.map(function (r) {
                    //         return r.Oil_Pred_ActualChoke;
                    //     }),
                    //     borderColor: COLORS.orange,
                    //     borderDash: [4, 4],
                    //     tension: 0.2,
                    // },
                    {
                        label: 'Oil prediction',
                        data: rows.map(function (r) {
                            return r.Oil_Pred_OptimalChoke;
                        }),
                        borderColor: COLORS.green,
                        tension: 0.2,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } },
                scales: { y: { beginAtZero: false } },
            },
        });
        chartWater = new Chart(waterCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Water aktual',
                        data: rows.map(function (r) {
                            return r.Water_Actual;
                        }),
                        borderColor: COLORS.blue,
                        tension: 0.2,
                    },
                    // {
                    //     label: 'Water pred (choke aktual)',
                    //     data: rows.map(function (r) {
                    //         return r.Water_Pred_ActualChoke;
                    //     }),
                    //     borderColor: COLORS.orange,
                    //     borderDash: [4, 4],
                    //     tension: 0.2,
                    // },
                    {
                        label: 'Water prediction',
                        data: rows.map(function (r) {
                            return r.Water_Pred_OptimalChoke;
                        }),
                        borderColor: COLORS.green,
                        tension: 0.2,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } },
                scales: { y: { beginAtZero: false } },
            },
        });
        chartChoke = new Chart(chokeCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Choke aktual (%)',
                        data: rows.map(function (r) {
                            return chokeScale(r.Choke_Aktual);
                        }),
                        borderColor: COLORS.blue,
                        tension: 0.2,
                    },
                    {
                        label: 'Choke rekomendasi (%)',
                        data: rows.map(function (r) {
                            return chokeScale(r.Choke_Rekomendasi);
                        }),
                        borderColor: COLORS.orange,
                        tension: 0.2,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } },
                scales: { y: { beginAtZero: false } },
            },
        });
    }

    function openModal() {
        document.getElementById('trainModal').classList.remove('hidden');
    }
    function closeModal() {
        document.getElementById('trainModal').classList.add('hidden');
    }

    function fmtNum(x, d) {
        d = d || 4;
        if (x == null || x === '' || (typeof x === 'number' && isNaN(x))) return '—';
        return Number(x).toFixed(d);
    }

    function renderSaveModalMetrics(m) {
        var box = document.getElementById('saveModelMetrics');
        if (!m) {
            box.innerHTML =
                '<p class="text-gray-500 text-sm" style="margin:0">Metrik tidak tersedia.</p>';
            return;
        }
        var algo = m.algorithm || '—';
        var folds = m.cv_folds != null ? String(m.cv_folds) : '—';
        box.innerHTML =
        '<div class="sm-title">Akurasi model</div>' +
        // '<p class="text-gray-600 text-sm" style="margin:0 0 0.5rem 0">Algoritma: <strong class="text-gray-800 font-semibold">' +
        // String(algo) +
        // '</strong> · CV folds: ' +
        // folds +
        // '</p>' +
        // '<div class="sm-row">' +
        // '<span class="label-oil">Oil (test)</span> R² ' +
        // fmtNum(m.test_r2_oil) +
        // ' · RMSE ' +
        // fmtNum(m.test_rmse_oil) +
        // '</div>' +
        // '<div class="sm-row">' +
        // '<span class="label-water">Water (test)</span> R² ' +
            // fmtNum(m.test_r2_water) +
            // ' · RMSE ' +
            // fmtNum(m.test_rmse_water) +
            // '</div>' +
            // '<div class="sm-title" style="margin-top:0.65rem">Validasi silang (R², train)</div>' +
            '<div class="sm-row">' +
                '<span class="label-oil">Oil</span> ' +
                'μ = ' + fmtNum(m.cv_r2_mean_oil * 100) + '% ' +
                '· σ = ' + fmtNum(m.cv_r2_std_oil) +
            '</div>' +
            '<div class="sm-row">' +
                '<span class="label-water">Water</span> ' +
                'μ = ' + fmtNum(m.cv_r2_mean_water * 100) + '% ' +
                '· σ = ' + fmtNum(m.cv_r2_std_water) +
            '</div>'
    }

    async function openSaveModelModal() {
        var inp = document.getElementById('modelNameInput');
        document.getElementById('saveModelMetrics').innerHTML =
            '<p class="text-gray-500 text-sm" style="margin:0">Memuat metrik…</p>';
        try {
            var st = await getJson('/api/train/state');
            renderSaveModalMetrics(st.metrics);
        } catch (e) {
            renderSaveModalMetrics(null);
        }
        if (!inp.value.trim() && wellNameForSave) {
            inp.value = String(wellNameForSave).replace(/\s+/g, '_');
        }
        document.getElementById('saveModelModal').classList.remove('hidden');
        inp.focus();
        inp.select();
    }

    function closeSaveModelModal() {
        document.getElementById('saveModelModal').classList.add('hidden');
    }

    async function refreshSaveModelButton() {
        try {
            var st = await getJson('/api/train/state');
            var ok = st.has_models && st.has_optimization;
            document.getElementById('btnOpenSaveModel').disabled = !ok;
        } catch (e) {
            document.getElementById('btnOpenSaveModel').disabled = true;
        }
    }

    function populateFeatureModal() {
        var box = document.getElementById('featureChecks');
        var toil = document.getElementById('targetOil');
        var twat = document.getElementById('targetWater');
        box.innerHTML = '';
        toil.innerHTML = '';
        twat.innerHTML = '';
        lastColumns.forEach(function (c) {
            var opt = document.createElement('option');
            opt.value = c;
            opt.textContent = c;
            toil.appendChild(opt.cloneNode(true));
            twat.appendChild(opt);
        });
        var tOil = 'BORE_OIL_VOL';
        var tWa = 'BORE_WAT_VOL';
        if (toil.querySelector('option[value="' + tOil + '"]')) toil.value = tOil;
        if (twat.querySelector('option[value="' + tWa + '"]')) twat.value = tWa;

        lastColumns.forEach(function (name) {
            var id = 'fc_' + name.replace(/[^a-zA-Z0-9]/g, '_');
            var lab = document.createElement('label');
            lab.className = 'modal-check-item';
            lab.setAttribute('for', id);
            var inp = document.createElement('input');
            inp.type = 'checkbox';
            inp.id = id;
            inp.value = name;
            if (DEFAULT_FEATURES.indexOf(name) >= 0) inp.checked = true;
            var span = document.createElement('span');
            span.className = 'modal-check-text';
            span.textContent = name;
            lab.appendChild(inp);
            lab.appendChild(span);
            box.appendChild(lab);
        });
    }

    async function refreshStateDates() {
        try {
            var st = await getJson('/api/train/state');
            var mn = st.test_date_min;
            var mx = st.test_date_max;
            var ds = document.getElementById('optDateStart');
            var de = document.getElementById('optDateEnd');
            if (mn && mx) {
                ds.disabled = false;
                de.disabled = false;
                ds.min = mn;
                ds.max = mx;
                de.min = mn;
                de.max = mx;
                document.getElementById('optDateHint').textContent =
                    'Rentang tanggal test: ' + mn + ' → ' + mx;
            }
        } catch (e) {
            /* ignore */
        }
    }

    function setSessionLabel(text) {
        var el = document.getElementById('sessionLabel');
        if (el) el.textContent = text;
    }

    function updateUploadButtonState() {
        var inp = document.getElementById('fileInput');
        var btn = document.getElementById('btnUpload');
        if (!btn || !inp) return;
        var hasFile = !!(inp.files && inp.files[0]);
        btn.disabled = !(sessionId && hasFile);
    }

    async function initSession() {
        try {
            var r = await fetch('/api/train/session', { method: 'POST' });
            var d = await r.json().catch(function () {
                return {};
            });
            if (d.ok && d.session_id) {
                sessionId = d.session_id;
                setSessionLabel('Sesi: ' + sessionId.slice(0, 10) + '…');
            } else {
                setSessionLabel('Gagal membuat sesi');
                toast('Gagal membuat sesi', true);
            }
        } catch (e) {
            setSessionLabel('Gagal membuat sesi');
            toast('Gagal membuat sesi: ' + (e.message || 'jaringan'), true);
        }
    }

    document.addEventListener('DOMContentLoaded', function () {
        initSession().then(function () {
            updateUploadButtonState();
            refreshSaveModelButton();
        });

        document.getElementById('fileInput').addEventListener('change', async function () {
            if (!sessionId) {
                await initSession();
            }
            updateUploadButtonState();
        });

        document.getElementById('btnUpload').addEventListener('click', async function () {
            var inp = document.getElementById('fileInput');
            if (!inp.files || !inp.files[0]) {
                toast('Pilih file dulu.', true);
                return;
            }
            if (!sessionId) {
                toast('Sesi belum siap.', true);
                return;
            }
            setBusy('step-upload', true);
            try {
                var fd = new FormData();
                fd.append('file', inp.files[0]);
                fd.append('session_id', sessionId);
                var r = await fetch('/api/train/upload', {
                    method: 'POST',
                    headers: { 'X-Session-Id': sessionId },
                    body: fd,
                });
                var d = await r.json();
                if (!r.ok || d.ok === false) throw new Error(d.error || 'Upload gagal');
                document.getElementById('uploadSummary').textContent =
                    d.n_rows + ' baris · ' + d.columns.length + ' kolom · ' + (d.n_wells || 0) + ' sumur';
                document.getElementById('btnPreprocess').disabled = false;
                toast('Unggah berhasil.');
            } catch (e) {
                toast(e.message, true);
            }
            setBusy('step-upload', false);
        });

        document.getElementById('btnPreprocess').addEventListener('click', async function () {
            setBusy('step-preprocess', true);
            try {
                var d = await postJson('/api/train/preprocess', {});
                var skip = d.skipped_rules_missing_columns || d.skipped_rules || [];
                document.getElementById('preSummary').textContent =
                    'Setelah filter: ' + d.n_rows + ' baris. Kolom aturan dilewati: ' + (skip.join(', ') || '—');
                var sel = document.getElementById('wellSelect');
                sel.innerHTML = '<option value="">— Pilih —</option>';
                (d.wells || []).forEach(function (w) {
                    var o = document.createElement('option');
                    o.value = w.well;
                    o.textContent = w.well + ' (' + w.count + ')';
                    sel.appendChild(o);
                });
                sel.disabled = false;
                document.getElementById('btnSelectWell').disabled = false;
                toast('Praproses selesai.');
            } catch (e) {
                toast(e.message, true);
            }
            setBusy('step-preprocess', false);
        });

        document.getElementById('btnSelectWell').addEventListener('click', async function () {
            var w = document.getElementById('wellSelect').value;
            if (!w) {
                toast('Pilih sumur.', true);
                return;
            }
            setBusy('step-well', true);
            try {
                var d = await postJson('/api/train/select-well', { well_name: w });
                wellNameForSave = w;
                lastColumns = d.columns || [];
                document.getElementById('wellSummary').textContent = d.n_rows + ' baris · ' + lastColumns.length + ' kolom';
                document.getElementById('btnOpenFeatures').disabled = false;
                populateFeatureModal();
                toast('Sumur dipilih.');
            } catch (e) {
                toast(e.message, true);
            }
            setBusy('step-well', false);
        });

        document.getElementById('btnOpenFeatures').addEventListener('click', function () {
            if (!lastColumns.length) {
                toast('Tidak ada kolom.', true);
                return;
            }
            populateFeatureModal();
            openModal();
        });

        document.getElementById('btnModalCancel').addEventListener('click', closeModal);
        document.getElementById('btnModalSave').addEventListener('click', async function () {
            var feats = [];
            document.querySelectorAll('#featureChecks input[type=checkbox]:checked').forEach(function (c) {
                feats.push(c.value);
            });
            if (!feats.length) {
                toast('Pilih minimal satu fitur.', true);
                return;
            }
            setBusy('step-feature', true);
            try {
                await postJson('/api/train/features', {
                    feature_columns: feats,
                    target_oil: document.getElementById('targetOil').value,
                    target_water: document.getElementById('targetWater').value,
                });
                document.getElementById('featureSummary').textContent =
                    feats.length + ' fitur · target oil/water diset.';
                document.getElementById('btnSplit').disabled = false;
                closeModal();
                toast('Fitur disimpan.');
            } catch (e) {
                toast(e.message, true);
            }
            setBusy('step-feature', false);
        });

        document.getElementById('btnSplit').addEventListener('click', async function () {
            setBusy('step-split', true);
            try {
                var d = await postJson('/api/train/split', {
                    test_size: parseFloat(document.getElementById('testSize').value),
                    random_state: parseInt(document.getElementById('randomState').value, 10),
                    cv_folds: parseInt(document.getElementById('cvFoldsSplit').value, 10),
                });
                document.getElementById('splitSummary').textContent =
                    'Train ' +
                    d.train_rows +
                    ' · Test ' +
                    d.test_rows +
                    '. Tanggal test: ' +
                    (d.test_date_min || '—') +
                    ' → ' +
                    (d.test_date_max || '—');
                document.getElementById('algoSelect').disabled = false;
                document.getElementById('btnTrain').disabled = false;
                await refreshStateDates();
                await refreshSaveModelButton();
                toast('Split OK.');
            } catch (e) {
                toast(e.message, true);
            }
            setBusy('step-split', false);
        });

        document.getElementById('btnTrain').addEventListener('click', async function () {
            var algo = document.getElementById('algoSelect').value;
            if (!algo) {
                toast('Pilih algoritma.', true);
                return;
            }
            setBusy('step-train', true);
            try {
                var d = await postJson('/api/train/train', { algorithm: algo });
                var ts =
                    'Test RMSE oil ' +
                    (d.test_rmse_oil != null ? Number(d.test_rmse_oil).toFixed(4) : '—') +
                    ' · R² ' +
                    (d.test_r2_oil != null ? Number(d.test_r2_oil).toFixed(4) : '—');
                document.getElementById('trainSummary').textContent = ts;
                document.getElementById('cvOil').textContent = fmtCv(d.cross_validation_oil);
                document.getElementById('cvWater').textContent = fmtCv(d.cross_validation_water);
                document.getElementById('btnCv').disabled = false;
                document.getElementById('btnOptimize').disabled = false;
                document.getElementById('btnOpenSaveModel').disabled = true;
                toast('Training selesai.');
            } catch (e) {
                toast(e.message, true);
            }
            setBusy('step-train', false);
        });

        document.getElementById('btnCv').addEventListener('click', async function () {
            setBusy('step-cv', true);
            try {
                var d = await postJson('/api/train/cross-validate', {
                    cv: parseInt(document.getElementById('cvFolds').value, 10),
                });
                document.getElementById('cvOil').textContent = fmtCv(d.cross_validation_oil);
                document.getElementById('cvWater').textContent = fmtCv(d.cross_validation_water);
                toast('CV diperbarui.');
            } catch (e) {
                toast(e.message, true);
            }
            setBusy('step-cv', false);
        });

        document.getElementById('btnOptimize').addEventListener('click', async function () {
            var ds = document.getElementById('optDateStart').value;
            var de = document.getElementById('optDateEnd').value;
            var body = {
                method: document.getElementById('optMethod').value,
                n_days: parseInt(document.getElementById('nDays').value, 10),
            };
            if (ds && de) {
                body.start_date = ds;
                body.end_date = de;
            }
            setBusy('step-optimize', true);
            try {
                var d = await postJson('/api/train/optimize', body);
                document.getElementById('optSummary').textContent =
                    d.n_rows + ' baris · subset: ' + JSON.stringify(d.subset || {});
                var rows = d.rows || [];
                buildChartsFromRows(rows);
                document.getElementById('btnOpenSaveModel').disabled =
                    !(rows.length && sessionId);
                toast('Optimasi selesai.');
            } catch (e) {
                toast(e.message, true);
            }
            setBusy('step-optimize', false);
        });

        document.getElementById('btnOpenSaveModel').addEventListener('click', async function () {
            if (document.getElementById('btnOpenSaveModel').disabled) return;
            await openSaveModelModal();
        });

        document.getElementById('btnSaveModelCancel').addEventListener('click', closeSaveModelModal);

        document.getElementById('btnSaveModelConfirm').addEventListener('click', async function () {
            var name = document.getElementById('modelNameInput').value.trim();
            if (!name) {
                toast('Isi nama model.', true);
                return;
            }
            try {
                var d = await postJson('/api/train/save-models', { model_name: name });
                document.getElementById('saveModelSummary').textContent =
                    'Tersimpan: ' + d.oil_path + ' · ' + d.water_path;
                closeSaveModelModal();
                toast('Model disimpan sebagai "' + (d.model_name || name) + '".');
            } catch (e) {
                toast(e.message, true);
            }
        });

        document.getElementById('saveModelModal').addEventListener('click', function (ev) {
            if (ev.target.id === 'saveModelModal') closeSaveModelModal();
        });

        document.getElementById('modelNameInput').addEventListener('keydown', function (ev) {
            if (ev.key === 'Enter') {
                ev.preventDefault();
                document.getElementById('btnSaveModelConfirm').click();
            }
        });
    });
})();
