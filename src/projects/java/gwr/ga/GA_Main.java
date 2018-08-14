package gwr.ga;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import ga.GeneticAlgorithm;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2D_Array;
import spawnn.som.grid.GridPos;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GA_Main {

	private static Logger log = Logger.getLogger(GA_Main.class);

	public static void main(String[] args) {

		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();
		GWKernel kernel = GWKernel.bisquare;
		boolean adaptive = true;
		double minBw = 8;
		int bwInit = -1;
		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 3 };
		int ta = 4;
		
		int dim = 50;
		Grid<double[]> grid = new Grid2D_Array<>(dim,dim);
		for( GridPos p : grid.getPositions() ) {
			double lon = p.getPos(0);
			double lat = p.getPos(1);
			
			double beta = 1;
			if( lon > dim/2 )
				beta = -1;
			double x1 = r.nextDouble();
			double y = x1 + beta;
						
			double[] d = new double[]{ lon, lat, beta, x1, y};
			grid.setPrototypeAt(p, d);
		}

		List<double[]> samples = new ArrayList<double[]>(grid.getPrototypes());
		List<Geometry> geoms = new ArrayList<Geometry>();
		for (double[] d : samples)
			geoms.add(gf.createPoint(new Coordinate(d[ga[0]], d[ga[1]])));
		log.debug(samples.size());
		DataUtils.writeCSV("output/spDat.csv", samples, new String[] { "long", "lat", "beta", "x1", "y" });

		Map<double[], Integer> idxMap = new HashMap<double[], Integer>();
		for (int i = 0; i < samples.size(); i++)
			idxMap.put(samples.get(i), i);
		
		Map<double[], Set<double[]>> cm = grid.getContiguityMap();
		Map<Integer, Set<Integer>> cmI = new HashMap<Integer, Set<Integer>>();		
		for( Entry<double[],Set<double[]>> e : cm.entrySet() ) {
			Set<Integer> s = new HashSet<>();
			for( double[] d : e.getValue() ) 
				s.add( samples.indexOf(d) );
			cmI.put(samples.indexOf(e.getKey()), s);
		}
		GWRIndividual.cmI = cmI;
		
		Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.contiguityMapToDistanceMap(cm));

		GWRCostCalculator ccAICc = new GWRIndividualCostCalculator_AICc(samples, fa, ga, ta, kernel, adaptive, minBw);

		{
			log.info("Search global bandwidth/j");
			double bestGlobalACIc = Double.MAX_VALUE;
			for (int j = 2; j < dim; j++) {
				List<Double> bw = new ArrayList<Double>();
				for (int i = 0; i < samples.size(); i++)
					bw.add((double) j);
				GWRIndividual ind = new GWRIndividual(bw, 0, Double.MAX_VALUE, 0);

				double aicc = ccAICc.getCost(ind);

				if (aicc < bestGlobalACIc) {
					bestGlobalACIc = aicc;
					bwInit = j;
					log.info("new best bw "+j+", " + aicc);
				}
			}
		}

		// best ts2: -242.13259853371318
		// best ts3: -222.98178679677832
		GeneticAlgorithm.tournamentSize = 2;
		GeneticAlgorithm.elitist = true;
		GeneticAlgorithm.recombProb = 0.7;
		GeneticAlgorithm.threads = 6;

		GWRIndividual.mutationMode = false;
		GWRIndividual.meanRecomb = false;
		double sd = -1; // only used if mutationMode==true
		boolean intInd = true;

		List<GWRIndividual> init = new ArrayList<GWRIndividual>();
		while (init.size() < 100) {
			List<Double> bandwidth = new ArrayList<>();
			while (bandwidth.size() < samples.size())
				bandwidth.add((double) Math.round(bwInit + r.nextGaussian() * 4));
			if (!intInd)
				init.add(new GWRIndividual(bandwidth, sd, bandwidth.size(), minBw));
			else
				init.add(new GWRIndividual_Int(bandwidth, sd, bandwidth.size(), minBw));
		}

		GeneticAlgorithm<GWRIndividual> gen = new GeneticAlgorithm<GWRIndividual>();
		GWRIndividual result = (GWRIndividual) gen.search(init, ccAICc);
		Map<double[], Double> resultBw = ccAICc.getSpatialBandwidth(result);

		double aicc = ccAICc.getCost(result);

		log.info("result GWR AICc: " + aicc);

		List<double[]> rr = new ArrayList<double[]>();
		Map<double[], Double> r2 = new HashMap<double[], Double>();
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			rr.add(new double[] { d[ga[0]], d[ga[1]], d[2], resultBw.get(d) });
			r2.put(d, resultBw.get(d));
		}
		log.info("moran: " + Arrays.toString(GeoUtils.getMoransIStatistics(dMap, r2)));

		String s = "output/result_AICc_" + aicc;
		DataUtils.writeCSV(s + ".csv", rr, new String[] { "long", "lat", "b", "radius" });
	}

}
