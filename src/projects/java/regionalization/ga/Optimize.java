package regionalization.ga;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.RegionUtils;

public class Optimize {

	private static Logger log = Logger.getLogger(Optimize.class);

	public static void main(String[] args) {
		int numRegions = 10;

		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/regionalization/200rand.shp"), new int[] {}, true);

		int[] fa = new int[] { 7 };

		for (int i : fa)
			DataUtils.zScoreColumn(samples, i);

		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/200rand.ctg");

		int runs = 200;

		for (boolean maio : new boolean[] { true, false }) {
			for (int mode : new int[] { 0, 1, 2, 3 }) {

				GeneticAlgorithm.mode = mode;
				DescriptiveStatistics ds = new DescriptiveStatistics();

				for (int i = 0; i < runs; i++) {
					List<GAIndividual> init = new ArrayList<GAIndividual>();
					for (int j = 0; j < 25; j++) {
						List<double[]> chromosome = new ArrayList<double[]>(samples);
						Collections.shuffle(chromosome);
						init.add(new RegioGAIndividual(chromosome, numRegions, new WCSSCostCalulator(new EuclideanDist(fa)), cm));
					}

					GeneticAlgorithm ga = new GeneticAlgorithm();

					RegioGAIndividual result = (RegioGAIndividual) ga.search(init);
					ds.addValue(result.getValue());
				}
				log.debug(maio + "," + mode + "," + ds.getMin() + "," + ds.getMean() + "," + ds.getMax() + "," + ds.getStandardDeviation());
			}
		}
	}
}
