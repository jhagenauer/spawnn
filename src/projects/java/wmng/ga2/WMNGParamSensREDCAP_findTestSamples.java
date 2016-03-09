package wmng.ga2;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.RegionUtils;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

public class WMNGParamSensREDCAP_findTestSamples {

	private static Logger log = Logger.getLogger(WMNGParamSensREDCAP_findTestSamples.class);

	public static void main(String[] args) {

		final int T_MAX = 150000;
		final int nrNeurons = 10;
		final Random r = new Random();

		File file = new File("data/redcap/Election/election2004.shp");
		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(file, new int[] {}, true);
		final List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(file);

		// build dist matrix and add coordinates to samples
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();

			// normed coordinates
			d[2] = p1.getX();
			d[3] = p1.getY();
		}

		final int[] ga = new int[] { 0, 1 };
		final int[] gaNormed = new int[] { 2, 3 };

		final int fa = 7; // bush pct
		final int fips = 4; // county_f basically identical to fips
		final Dist<double[]> fDist = new EuclideanDist(new int[] { fa });
		final Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.zScoreColumn(samples, fa);
		DataUtils.zScoreGeoColumns(samples, gaNormed, gDist);

		final Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");

		final Map<double[][],Integer> pairs = new HashMap<double[][],Integer>();
		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);

			if (a[fips] == 0)
				continue;

			double[] ma = DataUtils.getMean(ctg.get(a));
			for (int j = i + 1; j < samples.size() - 1; j++) {
				double[] b = samples.get(j);

				if (a == b || b[fips] == 0)
					continue;
				double[] mb = DataUtils.getMean(ctg.get(b));

				double ctxD = fDist.dist(ma, mb);
				
				if (ctxD > 2)
					pairs.put(new double[][] { a, b },0 );
			}
		}
		log.debug("pairs: "+pairs.size());

		log.debug("Training...");
		for( int k = 0; k < 500; k++ )
		for( int l = 1; l <= nrNeurons; l++ ) {
			log.debug("l: "+l);
			List<double[]> neurons = new ArrayList<double[]>();
			for (int i = 0; i < nrNeurons; i++) {
				double[] rs = samples.get(r.nextInt(samples.size()));
				neurons.add(Arrays.copyOf(rs, rs.length));
			}
	
			Sorter<double[]> bg = new KangasSorter<double[]>(gDist, fDist, l);
			NG ng = new NG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);
	
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				ng.train((double) t / T_MAX, x);
			}
	
			Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);		
			for( double[][] d : pairs.keySet() ) {
							
				boolean sameCluster = true;
				for( Set<double[]> s : bmus.values() )
					if( s.contains(d[0]) && !s.contains(d[1]) )
						sameCluster = false;
				if( !sameCluster )
					pairs.put(d, pairs.get(d)+1);
			}
		}
		
		List<double[][]> sl = new ArrayList<double[][]>(pairs.keySet());
		Collections.sort(sl,new Comparator<double[][]>() {
			@Override
			public int compare(double[][] o1, double[][] o2) {
				return Integer.compare(pairs.get(o1), pairs.get(o2));
			}
		});
		for( double[][] d : sl.subList(0, 20))
			log.debug((int)d[0][fips]+","+(int)d[1][fips]+","+pairs.get(d));
	}

	public static boolean sameCluster(Map<double[], Set<double[]>> bmus, double[][] samples) {
		for (Set<double[]> s : bmus.values()) {
			int count = 0;
			for (double[] a : s)
				for (double[] b : samples)
					if (a == b)
						count++;
			if (count == samples.length)
				return true;
		}
		return false;
	}

	public static double[] getSampleByFips(List<double[]> samples, int fc, int fips) {
		for (double[] d : samples)
			if (d[fc] == fips)
				return d;
		return null;
	}
}
