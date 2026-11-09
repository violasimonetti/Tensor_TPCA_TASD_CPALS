function plot_tensor_simulations(results, outputDir)
%PLOT_TENSOR_SIMULATIONS Figures for the three evaluation classes in v3.
%
% Dashed curves show initialization; solid curves show the final CP-ALS fit.
% Unequal-rank panels use matched error and component F1 explicitly.

    if results.completedConditions ~= numel(results.conditions)
        error('plot_tensor_simulations:IncompleteRun', ...
            'Finish all conditions before exporting the report figures.');
    end
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end
    summary = results.summaryTable;

    plot_panels(summary, 'meanFinalError', 'Mean worst loading error', ...
        'Reconstruction performance: loading recovery', ...
        'meanInitialError', 'meanFinalMatchedError', ...
        'Matched loading error (partial recovery)', ...
        'meanInitialMatchedError', 'loading_recovery', outputDir, false);

    plot_panels(summary, 'meanTotalSeconds', 'Mean total time (seconds)', ...
        'Optimization behaviour', '', '', '', '', ...
        'optimization_behaviour', outputDir, false);

    plot_panels(summary, 'recoveryProbability', 'Probability of full loading recovery', ...
        'Stability and robustness', '', 'meanFinalF1', ...
        'Mean component recovery F1', '', ...
        'stability_robustness', outputDir, true);

    plot_panels(summary, 'meanALSItersCompleted', 'Mean ALS iterations (completed calls)', ...
        'Optimization behaviour: ALS iterations', '', '', '', '', ...
        'als_iterations', outputDir, false);

    plot_panels(summary, 'meanReconError', 'Mean relative tensor reconstruction error', ...
        'Supporting tensor reconstruction accuracy', '', '', '', '', ...
        'tensor_reconstruction', outputDir, false);

    if results.config.nStarts > 1
        plot_panels(summary, 'meanWithinDataDistance', 'Mean distance between repeated fits', ...
            'Stability on the same observed tensor', '', '', '', '', ...
            'initialization_stability', outputDir, false);
    end

    plot_rank_recovery(summary, outputDir);

end

function plot_panels(summary, finalField, yLabel, figureTitle, ...
        initialField, rankFinalField, rankLabel, rankInitialField, ...
        fileName, outputDir, probabilityAxis)

    studies = ["noise_sigma", "factor_strength", "coherence", "rank_misspecification"];
    methods = ["TASD", "TPCA"];
    colors = [0.0000 0.4470 0.7410; 0.8500 0.3250 0.0980];
    fig = figure('Name', figureTitle, 'Color', 'w', ...
        'Visible', 'off', 'Position', [100 100 1100 720]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for s = 1:numel(studies)
        nexttile;
        hold on;
        field = finalField;
        beforeField = initialField;
        panelLabel = yLabel;
        if studies(s) == "rank_misspecification" && ~isempty(rankFinalField)
            field = rankFinalField;
            beforeField = rankInitialField;
            panelLabel = rankLabel;
        end

        for m = 1:numel(methods)
            sub = summary(summary.study == studies(s) & summary.method == methods(m), :);
            [sub, x, panelTitle] = panel_coordinates(sub, studies(s));
            if ~isempty(beforeField)
                plot(x, sub.(beforeField), '--s', 'Color', colors(m, :), ...
                    'LineWidth', 1.2, 'DisplayName', methods(m) + " initial");
                finalName = methods(m) + " final";
            else
                finalName = methods(m);
            end
            plot(x, sub.(field), '-o', 'Color', colors(m, :), ...
                'LineWidth', 1.5, 'MarkerSize', 5, 'DisplayName', finalName);
        end

        % Error curves can span many orders of magnitude in the noise study.
        if contains(field, 'Error')
            values = summary.(field)(summary.study == studies(s));
            if ~isempty(beforeField)
                values = [values; summary.(beforeField)(summary.study == studies(s))];
            end
            if all(isfinite(values) & values > 0)
                set(gca, 'YScale', 'log');
            end
        end
        if probabilityAxis
            ylim([0 1]);
        end
        ylabel(panelLabel);
        title(panelTitle);
        grid on;
        box on;
        legend('Location', 'best');
    end

    sgtitle(figureTitle);
    save_figure(fig, outputDir, fileName);

end

function [sub, x, panelTitle] = panel_coordinates(sub, study)

    switch study
        case "noise_sigma"
            sub = sortrows(sub, 'sigma_used');
            x = sub.sigma_used;
            set(gca, 'XScale', 'log');
            xlabel('Noise standard deviation \sigma');
            panelTitle = 'Noise level';
        case "factor_strength"
            x = (1:height(sub))';
            xticks(x);
            xticklabels(sub.label);
            xlabel('Component strengths');
            panelTitle = 'Factor strength';
        case "coherence"
            sub = sortrows(sub, 'xi_used');
            x = sub.xi_used;
            xlabel('Column coherence \xi');
            panelTitle = 'Coherence';
        case "rank_misspecification"
            sub = sortrows(sub, 'Rfit_used');
            x = sub.Rfit_used;
            xticks(x);
            xlabel('Fitted CP rank');
            panelTitle = 'Rank misspecification';
    end

end

function plot_rank_recovery(summary, outputDir)

    methods = ["TASD", "TPCA"];
    colors = [0.0000 0.4470 0.7410; 0.8500 0.3250 0.0980];
    fields = ["meanFinalRecall", "meanFinalPrecision"];
    labels = ["Mean component recall", "Mean component precision"];
    fig = figure('Name', 'Rank misspecification: component recovery', ...
        'Color', 'w', 'Visible', 'off', 'Position', [100 100 1100 400]);
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    for panel = 1:2
        nexttile;
        hold on;
        for m = 1:numel(methods)
            sub = summary(summary.study == "rank_misspecification" & ...
                summary.method == methods(m), :);
            sub = sortrows(sub, 'Rfit_used');
            plot(sub.Rfit_used, sub.(fields(panel)), '-o', ...
                'Color', colors(m, :), 'LineWidth', 1.5, ...
                'DisplayName', methods(m));
        end
        xticks(sub.Rfit_used);
        xlabel('Fitted CP rank');
        ylabel(labels(panel));
        ylim([0 1]);
        grid on;
        box on;
        legend('Location', 'best');
    end
    sgtitle('Rank misspecification: component recovery');
    save_figure(fig, outputDir, 'rank_recovery');

end

function save_figure(fig, outputDir, fileName)

    drawnow;
    exportgraphics(fig, fullfile(outputDir, [fileName '.pdf']), 'ContentType', 'vector');
    exportgraphics(fig, fullfile(outputDir, [fileName '.png']), 'Resolution', 180);
    close(fig);

end
