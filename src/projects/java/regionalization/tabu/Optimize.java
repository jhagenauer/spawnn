package regionalization.tabu;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;

import regionalization.RegionUtils;
import regionalization.ga.InequalityCalculator;
import spawnn.utils.DataUtils;

public class Optimize {

	private static Logger log = Logger.getLogger(Optimize.class);

	public static void main(String[] args) {
		final int numRegions = 7;

		/*
		 * final List<double[]> samples = DataUtil.readSamplesFromShapeFile(new
		 * File("data/regionalization/100rand.shp"), new int[] {}, true); final
		 * int[] fa = new int[] { 7 }; for (int i : fa)
		 * DataUtil.zScoreColumn(samples, i); final Map<double[], Set<double[]>>
		 * cm = RegionUtils.readContiguitiyMap(samples,
		 * "data/regionalization/100rand.ctg");
		 */

		// best results 25, 60, 50, 90 (4985)
		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/lisbon/lisbon.shp"), new int[] {}, true);
		final int[] fa = new int[] { 1 };
		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/lisbon/lisbon_queen.ctg");

		double mean = 0;
		for (double[] d : samples)
			mean += d[fa[0]] / numRegions;
		final double MEAN = mean;

		int runs = 1;

		for (int tlLength = 5; tlLength <= 20; tlLength += 5) {
			for (int f = 10; f <= 100; f+=10) {
				for (int dur = 10; dur <= 60; dur += 10) {
					for (int s = dur; s <= 100; s += 10) {

						ExecutorService es = Executors.newFixedThreadPool(10);
						List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
						
						final int l = tlLength;
						final int F = f;
						final int DUR = dur;
						final int S = s;

						for (int r = 0; r < runs; r++) {
							futures.add(es.submit(new Callable<double[]>() {

								@Override
								public double[] call() throws Exception {

									List<double[]> chrom = new ArrayList<double[]>(samples);
									Collections.shuffle(chrom);
									RegioTabuIndividual init = new RegioTabuIndividual(chrom, numRegions, new InequalityCalculator(fa, MEAN), cm );
									RegioTabuSearch ts = new RegioTabuSearch();
									ts.tlLength = l;
									ts.f = F;
									ts.penaltyDur = DUR;
									ts.penaltyStart = S;
									
									TabuIndividual result = (TabuIndividual) ts.search(init);
									// log.debug(result.getValue());
									return new double[] { result.getValue() };
								}
							}));
						}
						es.shutdown();

						double avg = 0;
						for (Future<double[]> fu : futures) {
							try {
								avg += fu.get()[0]/runs;
							} catch (InterruptedException e) {
								e.printStackTrace();
							} catch (ExecutionException e) {
								e.printStackTrace();
							}
						}
						log.debug(l + ", " +F+", "+DUR+", "+S+": " + avg);
					}
				}
			}
		}
	}
}
