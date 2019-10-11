// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class AnomalyDetectionMetricsAgent : IMetricsAgent<AnomalyDetectionMetrics>
    {
        private readonly MLContext _mlContext;
        private readonly AnomalyDetectionMetric _optimizingMetric;

        public AnomalyDetectionMetricsAgent(MLContext mlContext,
            AnomalyDetectionMetric optimizingMetric)
        {
            _mlContext = mlContext;
            _optimizingMetric = optimizingMetric;
        }

        public double GetScore(AnomalyDetectionMetrics metrics)
        {
            if (metrics == null)
            {
                return double.NaN;
            }

            switch (_optimizingMetric)
            {
                case AnomalyDetectionMetric.AreaUnderRocCurve:
                    return metrics.AreaUnderRocCurve;
                case AnomalyDetectionMetric.DetectionRateAtFalsePositiveCount:
                    return metrics.DetectionRateAtFalsePositiveCount;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public bool IsModelPerfect(double score)
        {
            if (double.IsNaN(score))
            {
                return false;
            }

            switch (_optimizingMetric)
            {
                case AnomalyDetectionMetric.AreaUnderRocCurve:
                    return score == 1;
                case AnomalyDetectionMetric.DetectionRateAtFalsePositiveCount:
                    return score == 1; // Is this really correct?
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public AnomalyDetectionMetrics EvaluateMetrics(IDataView data, string labelColumn)
        {
            return _mlContext.AnomalyDetection.Evaluate(data, labelColumn);
        }
    }
}
