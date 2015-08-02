package regionalization.ga;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import regionalization.RegionUtils;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;

public class TestRegioGA {

	private static Logger log = Logger.getLogger(TestRegioGA.class);

	public static void main(String[] args) {
		int numRegions = 10;

		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/regionalization/200rand.shp"), new int[] {}, true);

		int[] fa = new int[] { 7 };

		for (int i : fa)
			DataUtils.zScoreColumn(samples, i);

		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/200rand.ctg");

		int runs = 10;
		
		{
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (int i = 0; i < runs; i++) {
			log.info("run: "+i);
			
			List<GAIndividual> init = new ArrayList<GAIndividual>();
			for (int j = 0; j < 25; j++) {
				List<double[]> chromosome = new ArrayList<double[]>(samples);
				Collections.shuffle(chromosome);
				init.add(new RegioGAIndividual(chromosome, numRegions, new WCSSCostCalulator(new EuclideanDist(fa)), cm));	
			}
			GeneticAlgorithm ga = new GeneticAlgorithm();
			GAIndividual result = (GAIndividual) ga.search(init);
			
			ds.addValue( result.getValue() );
			
		}
		log.info("value: "+ds.getMin()+","+ds.getMean()+","+ds.getMax()+","+ds.getStandardDeviation() );
		}
	}
}
