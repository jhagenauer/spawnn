package context.time;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterMNG;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.GridPos;
import spawnn.utils.DataUtils;

public class TimeSeriesMNGOptimize {

	private static Logger log = Logger.getLogger(TimeSeriesMNGOptimize.class);

	public static void main(String[] args) {

		try {

			final Dist<double[]> fDist = new EuclideanDist(new int[] { 1 });
			// Dist tDist = new SubDist(eDist, new int[]{0});

			final List<double[]> samples = DataUtils.readCSV( new FileInputStream("data/mg/mgsamples.csv")).subList(0, 150000);
			final int T_MAX = samples.size();
			final int rcpFieldSize = 30;

			ExecutorService es = Executors.newFixedThreadPool(8);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (double alpha = 0; alpha < 1.0; alpha += 0.01) {
				for( final double BETA : new double[]{ 0.75 } ) {

				final double ALPHA = alpha;

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						SummaryStatistics ss = new SummaryStatistics();

						for (int i = 0; i < 8; i++) {

							SorterMNG bg = new SorterMNG(fDist, ALPHA,BETA);
							ContextNG ng = new ContextNG(100, 50.0, 0.01, 0.5, 0.005, 4, bg);
							
							DecayFunction df = new LinearDecay(1.0, ALPHA );

							for (int t = 0; t < T_MAX; t++) {
								
								bg.setAlpha( df.getValue( (double) t / T_MAX ) );
								
								double[] x = samples.get(t % samples.size());
								ng.train((double) t / T_MAX, x);
							}

							bg.setLastBmu(null);

							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
							double[] tqe = TimeSeries.getTemporalQuantizationError(samples,	bmus, fDist, rcpFieldSize);

							double sum = 0;
							for (double d : tqe)
								sum += d;
							ss.addValue(sum);

						}

						double[] d = new double[] { ss.getMean(), ALPHA, BETA };
						log.info(Arrays.toString(d));

						return d;

					}
				}));

			}}

			es.shutdown();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public static Map<GridPos, List<List<double[]>>> getReceptiveField(
			List<double[]> samples, Map<GridPos, Set<double[]>> bmus,
			int rcpFieldSize) {
		Map<GridPos, List<List<double[]>>> bmuSeqs = new HashMap<GridPos, List<List<double[]>>>();
		for (int i = rcpFieldSize - 1; i < samples.size(); i++) {

			double[] x = samples.get(i);
			GridPos bmu = null;
			for (GridPos gp : bmus.keySet())
				if (bmus.get(gp).contains(x))
					bmu = gp;

			if (!bmuSeqs.containsKey(bmu))
				bmuSeqs.put(bmu, new ArrayList<List<double[]>>());

			List<double[]> sub = samples.subList(i - rcpFieldSize + 1, i + 1);
			bmuSeqs.get(bmu).add(sub);
		}
		return bmuSeqs;
	}

	public static Map<GridPos, List<double[]>> getMeanReceptiveField(
			Map<GridPos, List<List<double[]>>> bmuSeqs) {
		int inputDim = -1;
		int fieldLength = -1;
		for (List<List<double[]>> l1 : bmuSeqs.values()) {
			for (List<double[]> l2 : l1) {

				if (fieldLength > 0 && fieldLength != l2.size()) {
					return null;
				} else if (fieldLength < 0)
					fieldLength = l2.size();

				if (!l2.isEmpty()) {
					inputDim = l2.get(0).length;
					break;
				}
				if (inputDim > 0)
					break;
			}
		}

		Map<GridPos, List<double[]>> meanRcpFields = new HashMap<GridPos, List<double[]>>();
		for (GridPos bmu : bmuSeqs.keySet()) {

			List<double[]> meanList = new ArrayList<double[]>();
			for (int i = 0; i < fieldLength; i++) {

				double[] d = new double[inputDim];
				for (List<double[]> l : bmuSeqs.get(bmu))
					for (int j = 0; j < d.length; j++)
						d[j] += l.get(i)[j] / bmuSeqs.get(bmu).size();

				meanList.add(d);
			}
			meanRcpFields.put(bmu, meanList);
		}
		return meanRcpFields;
	}

	public static double[] getTemporalQuantizationError(List<double[]> samples,
			Map<GridPos, Set<double[]>> bmus, Dist<double[]> dist,
			int rcpFieldSize) {
		Map<GridPos, List<List<double[]>>> rcpFields = getReceptiveField(
				samples, bmus, rcpFieldSize);
		Map<GridPos, List<double[]>> meanRcpFields = getMeanReceptiveField(rcpFields);

		double[] tqe = new double[rcpFieldSize];
		for (int i = 0; i < rcpFieldSize; i++) {
			double sum = 0;

			int k = 0;
			for (int j = rcpFieldSize - 1; j < samples.size(); j++) {

				double[] x = samples.get(j);
				GridPos bmu = null;
				for (GridPos gp : bmus.keySet())
					if (bmus.get(gp).contains(x))
						bmu = gp;

				List<double[]> meanSeq = meanRcpFields.get(bmu);

				sum += Math.pow(
						dist.dist(samples.get(j - i),
								meanSeq.get(meanSeq.size() - 1 - i)), 2);

				k++;
			}
			tqe[i] = Math.sqrt(sum / k);
		}
		return tqe;
	}
}
