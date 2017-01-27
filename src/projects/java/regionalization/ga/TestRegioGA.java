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
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class TestRegioGA {

	private static Logger log = Logger.getLogger(TestRegioGA.class);

	public static void main(String[] args) {
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/philadelphia/data_philly.shp"), true);
		List<double[]> samples = sdf.samples;
		int[] fa = new int[] { 3 /*, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13*/ };
		DataUtils.transform(samples, fa, Transform.zScore);

		final Map<double[], Set<double[]>> cm = GeoUtils.getContiguityMap(samples, sdf.geoms, false, false);

		int numRegions = 10;
		int runs = 10;

		{
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i = 0; i < runs; i++) {
				log.info("run: " + i);

				List<GAIndividual> init = new ArrayList<GAIndividual>();
				for (int j = 0; j < 25; j++) {
					List<double[]> chromosome = new ArrayList<double[]>(samples);
					Collections.shuffle(chromosome);
					init.add(new RegioGAIndividual(chromosome, numRegions, new WCSSCostCalulator(new EuclideanDist(fa)), cm));
				}
				GeneticAlgorithm ga = new GeneticAlgorithm();
				GAIndividual result = (GAIndividual) ga.search(init);

				ds.addValue(result.getValue());

			}
			log.info("value: " + ds.getMin() + "," + ds.getMean() + "," + ds.getMax() + "," + ds.getStandardDeviation());
		}
	}
}
