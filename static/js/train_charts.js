/**
 * D3 charts for Train Model: scatter (actual vs predicted) and line (index-ordered).
 */

(function (global) {
    'use strict';

    function drawTrainScatterActualPred(containerId, opts) {
        var actual = opts.actual;
        var predicted = opts.predicted;
        var title = opts.title || '';
        var subtitle = opts.subtitle || '';
        var color = opts.color || '#4a90e2';

        var container = document.getElementById(containerId);
        if (!container || !actual || !predicted || !actual.length) return;
        container.innerHTML = '';

        var data = actual.map(function (a, i) {
            return { x: a, y: predicted[i] };
        });
        var all = actual.concat(predicted);
        var lo = d3.min(all);
        var hi = d3.max(all);
        var span = hi - lo;
        var pad = span > 0 ? span * 0.06 : Math.abs(lo || 1) * 0.06;

        var margin = { top: 44, right: 20, bottom: 44, left: 52 };
        var width = Math.min(container.clientWidth || 480, 640);
        var height = 280;
        var innerW = width - margin.left - margin.right;
        var innerH = height - margin.top - margin.bottom;

        var svg = d3.select('#' + containerId)
            .append('svg')
            .attr('class', 'd3-chart')
            .attr('width', width)
            .attr('height', height);

        svg.append('text')
            .attr('x', margin.left)
            .attr('y', 22)
            .attr('fill', '#1e293b')
            .attr('font-size', 13)
            .attr('font-weight', 600)
            .text(title);

        svg.append('text')
            .attr('x', margin.left)
            .attr('y', 38)
            .attr('fill', '#64748b')
            .attr('font-size', 11)
            .text(subtitle);

        var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

        var xScale = d3.scaleLinear()
            .domain([lo - pad, hi + pad])
            .nice()
            .range([0, innerW]);
        var yScale = d3.scaleLinear()
            .domain([lo - pad, hi + pad])
            .nice()
            .range([innerH, 0]);

        var d0 = lo - pad;
        var d1 = hi + pad;
        g.append('line')
            .attr('x1', xScale(d0))
            .attr('y1', yScale(d0))
            .attr('x2', xScale(d1))
            .attr('y2', yScale(d1))
            .attr('stroke', '#cbd5e1')
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '5,5');

        g.selectAll('.dot')
            .data(data)
            .enter()
            .append('circle')
            .attr('cx', function (d) { return xScale(d.x); })
            .attr('cy', function (d) { return yScale(d.y); })
            .attr('r', 4)
            .attr('fill', color)
            .attr('fill-opacity', 0.7)
            .attr('stroke', '#fff')
            .attr('stroke-width', 1);

        g.append('g')
            .attr('transform', 'translate(0,' + innerH + ')')
            .call(d3.axisBottom(xScale).ticks(6))
            .call(function (sel) { sel.selectAll('text').attr('fill', '#64748b'); });

        g.append('g')
            .call(d3.axisLeft(yScale).ticks(6))
            .call(function (sel) { sel.selectAll('text').attr('fill', '#64748b'); });

        g.append('text')
            .attr('x', innerW / 2)
            .attr('y', innerH + 36)
            .attr('text-anchor', 'middle')
            .attr('fill', '#64748b')
            .attr('font-size', 11)
            .text('Aktual');

        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -innerH / 2)
            .attr('y', -40)
            .attr('text-anchor', 'middle')
            .attr('fill', '#64748b')
            .attr('font-size', 11)
            .text('Prediksi');
    }

    function drawTrainDualLine(containerId, opts) {
        var actual = opts.actual;
        var predicted = opts.predicted;
        var title = opts.title || '';
        var colorA = opts.colorA || '#0f766e';
        var colorP = opts.colorP || '#14b8a6';

        var container = document.getElementById(containerId);
        if (!container || !actual || !predicted || !actual.length) return;
        container.innerHTML = '';

        var n = actual.length;
        if (n < 2) {
            container.innerHTML = '<p class="text-sm text-gray-500 text-center py-10 px-4">Grafik deret membutuhkan minimal 2 titik pada sample test.</p>';
            return;
        }
        var seriesA = d3.range(n).map(function (i) {
            return { x: i, y: actual[i] };
        });
        var seriesP = d3.range(n).map(function (i) {
            return { x: i, y: predicted[i] };
        });

        var margin = { top: 40, right: 24, bottom: 40, left: 52 };
        var width = Math.min(container.clientWidth || 480, 640);
        var height = 260;
        var innerW = width - margin.left - margin.right;
        var innerH = height - margin.top - margin.bottom;

        var allY = actual.concat(predicted);
        var yMin = d3.min(allY);
        var yMax = d3.max(allY);
        var yPad = (yMax - yMin) * 0.08 || 1;

        var svg = d3.select('#' + containerId)
            .append('svg')
            .attr('class', 'd3-chart')
            .attr('width', width)
            .attr('height', height);

        svg.append('text')
            .attr('x', margin.left)
            .attr('y', 22)
            .attr('fill', '#1e293b')
            .attr('font-size', 13)
            .attr('font-weight', 600)
            .text(title);

        var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

        var xScale = d3.scaleLinear().domain([0, n - 1]).range([0, innerW]);
        var yScale = d3.scaleLinear()
            .domain([yMin - yPad, yMax + yPad])
            .nice()
            .range([innerH, 0]);

        var line = d3.line()
            .x(function (d) { return xScale(d.x); })
            .y(function (d) { return yScale(d.y); })
            .curve(d3.curveMonotoneX);

        g.append('path')
            .datum(seriesA)
            .attr('fill', 'none')
            .attr('stroke', colorA)
            .attr('stroke-width', 2)
            .attr('d', line);

        g.append('path')
            .datum(seriesP)
            .attr('fill', 'none')
            .attr('stroke', colorP)
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '6,4')
            .attr('d', line);

        g.append('g')
            .attr('transform', 'translate(0,' + innerH + ')')
            .call(d3.axisBottom(xScale).ticks(Math.min(8, n)))
            .call(function (sel) { sel.selectAll('text').attr('fill', '#64748b'); });

        g.append('g')
            .call(d3.axisLeft(yScale).ticks(6))
            .call(function (sel) { sel.selectAll('text').attr('fill', '#64748b'); });

        g.append('text')
            .attr('x', innerW / 2)
            .attr('y', innerH + 32)
            .attr('text-anchor', 'middle')
            .attr('fill', '#64748b')
            .attr('font-size', 11)
            .text('Urutan titik (sample test)');

        var leg = g.append('g').attr('transform', 'translate(' + (innerW - 120) + ',0)');
        leg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 0).attr('y2', 0).attr('stroke', colorA).attr('stroke-width', 2);
        leg.append('text').attr('x', 24).attr('y', 4).attr('fill', '#475569').attr('font-size', 10).text('Aktual');
        leg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 14).attr('y2', 14).attr('stroke', colorP).attr('stroke-width', 2).attr('stroke-dasharray', '4,3');
        leg.append('text').attr('x', 24).attr('y', 18).attr('fill', '#475569').attr('font-size', 10).text('Prediksi');
    }

    function renderTrainVisualization(viz) {
        if (!viz || !viz.oil || !viz.water) return;

        drawTrainScatterActualPred('trainVizOilScatter', {
            title: 'Oil — Aktual vs Prediksi (test set)',
            subtitle: viz.target_oil || '',
            actual: viz.oil.actual,
            predicted: viz.oil.predicted,
            color: '#4a90e2'
        });
        drawTrainScatterActualPred('trainVizWaterScatter', {
            title: 'Water — Aktual vs Prediksi (test set)',
            subtitle: viz.target_water || '',
            actual: viz.water.actual,
            predicted: viz.water.predicted,
            color: '#ff7a5c'
        });
        drawTrainDualLine('trainVizOilLine', {
            title: 'Oil — perbandingan deret (sample)',
            actual: viz.oil.actual,
            predicted: viz.oil.predicted,
            colorA: '#2563eb',
            colorP: '#93c5fd'
        });
        drawTrainDualLine('trainVizWaterLine', {
            title: 'Water — perbandingan deret (sample)',
            actual: viz.water.actual,
            predicted: viz.water.predicted,
            colorA: '#c2410c',
            colorP: '#fdba74'
        });

        var cap = document.getElementById('vizCaption');
        if (cap) {
            cap.textContent =
                'Menampilkan ' + viz.sampled_points + ' titik dari ' + viz.total_test_points + ' baris test (scatter: garis putus = y = x ideal).';
        }
    }

    global.renderTrainVisualization = renderTrainVisualization;
})(typeof window !== 'undefined' ? window : this);
