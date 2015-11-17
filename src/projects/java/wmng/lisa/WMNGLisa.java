package wmng.lisa;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class WMNGLisa {

	private static Logger log = Logger.getLogger(WMNGLisa.class);

	public static void main(String[] args) {

		final int T_MAX = 500000;
		final int nrNeurons = 8;
		final Random r = new Random();

		File file = new File("data/redcap/Election/election2004.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(file, true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;

		final int fa = 7; // bush pct
		DataUtils.zScoreColumn(samples, fa);

		final Dist<double[]> fDist = new EuclideanDist(new int[] { fa });
		double alpha = 0.0;
		double beta = 0.0;

		final Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getContiguityMap(samples, geoms, false, false)));
		
		// ------------------------------------------------------------------------

		List<double[]> lisa = GeoUtils.getLocalMoransIMonteCarlo(samples, dMap, fa, 10000);
		Drawer.geoDrawValues(geoms, lisa, 0, sdf.crs, ColorMode.Blues, "output/lisa_mc.png");

		List<Double> values = new ArrayList<Double>();
		for (double[] d : lisa)
			if (d[4] < 0.0001)
				values.add(0.0);
			else if (d[4] < 0.001)
				values.add(1.0);
			else if (d[4] < 0.01)
				values.add(2.0);
			else if (d[4] < 0.05)
				values.add(3.0);
			else
				values.add(4.0);
		Drawer.geoDrawValues(geoms, values, sdf.crs, ColorMode.Spectral, "output/lisa_mc_signf.png");

		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (double[] d : samples)
			ds.addValue(d[fa]);
		double mean = ds.getMean();

		Map<Integer, Set<double[]>> lisaCluster = new HashMap<Integer, Set<double[]>>();
		for (int i = 0; i < samples.size(); i++) {
			double[] l = lisa.get(i);
			double[] d = samples.get(i);
			int clust = -1;

			if (l[4] > 0.05) // not significant
				clust = 0;
			else if (l[0] > 0 && d[fa] > mean)
				clust = 1; // high-high
			else if (l[0] > 0 && d[fa] < mean)
				clust = 2; // low-low
			else if (l[0] < 0 && d[fa] > mean)
				clust = 3; // high-low
			else if (l[0] < 0 && d[fa] < mean)
				clust = 4; // low-high
			else
				clust = 5; // unknown
			
			if (!lisaCluster.containsKey(clust))
				lisaCluster.put(clust, new HashSet<double[]>());
			lisaCluster.get(clust).add(d);
		}
		Drawer.geoDrawCluster(lisaCluster.values(), samples, geoms, "output/lisa_mc_clust.png", false);
		
		for( Entry<Integer,Set<double[]>> e : lisaCluster.entrySet() )
			log.debug(e.getKey()+":"+e.getValue().size());
	
		// -----------------------------------------------------------------------------------------
		
		List<double[]> neurons = new ArrayList<double[]>();
		for (int i = 0; i < nrNeurons; i++) {
			double[] rs = samples.get(r.nextInt(samples.size()));
			double[] d = Arrays.copyOf(rs, rs.length * 2);
			for (int j = rs.length; j < d.length; j++)
				d[j] = r.nextDouble();
			neurons.add(d);
		}

		Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
		for (double[] d : samples)
			bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

		SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, alpha, beta);
		DecayFunction nbRate = new PowerDecay((double) neurons.size() / 2, 0.01);
		DecayFunction adaptRate = new PowerDecay(0.6, 0.01);
		ContextNG ng = new ContextNG(neurons, nbRate, adaptRate, bg);

		bg.bmuHistMutable = true;
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));
			ng.train((double) t / T_MAX, x);
		}
		bg.bmuHistMutable = false;
	}
}
