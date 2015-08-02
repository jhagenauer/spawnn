package regionalization.zd;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import regionalization.RegionUtils;
import regionalization.ga.ClusterCostCalculator;
import regionalization.ga.GAIndividual;
import regionalization.ga.GeneticAlgorithm;
import regionalization.ga.InequalityCalculator;
import regionalization.ga.RegioGAIndividual;
import spawnn.utils.DataUtils;

public class ZDGeneticAlgorithm {
	
	private static Logger log = Logger.getLogger(ZDGeneticAlgorithm.class);
	int threads = 16;
		
	public static void main(String[] args) {
		int numRegions = 7;
				
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/lisbon/lisbon.shp"), new int[] {}, true);
		int[] fa = new int[] { 1 };
		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/lisbon/lisbon_queen.ctg");
		
		double mean = 0;
		for( double[] d : samples )
			mean += d[fa[0]]/numRegions;
		
		log.debug("mean: "+mean);
		
		ClusterCostCalculator cc = new InequalityCalculator(fa, mean);
			
		List<GAIndividual> init = new ArrayList<GAIndividual>();
		for( int i = 0; i < 50; i++ ) {
			List<double[]> chromosome = new ArrayList<double[]>(samples);
			Collections.shuffle(chromosome);
			init.add( new RegioGAIndividual( chromosome, numRegions, cc , cm ) );
		}
		
		GeneticAlgorithm ga = new GeneticAlgorithm();
		GAIndividual result = (GAIndividual)ga.search( init );
		
		for( Set<double[]> s : ((RegioGAIndividual)result).getCluster() )
			log.debug(cc.getCost(s));
		
	}
}
