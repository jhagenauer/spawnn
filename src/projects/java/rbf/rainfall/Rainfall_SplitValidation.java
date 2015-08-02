package rbf.rainfall;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.rbf.RBF;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;

public class Rainfall_SplitValidation {

	private static Logger log = Logger.getLogger(Rainfall_SplitValidation.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 25000;

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/rainfall_tw/rainfall_tw.csv"), new int[] {}, true);
		Collections.shuffle(df.samples);

		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();

		for (double[] d : df.samples) {
			samples.add(Arrays.copyOfRange(d, 1, d.length - 1));
			desired.add(new double[] { d[d.length - 1] });
		}
		int[] fa1 = new int[] { 0, 1 };

		// DataUtils.zScoreColumns(samples, fa1 );
		// DataUtils.normalizeColumns(samples, fa1 );

		final Dist<double[]> fDist = new EuclideanDist(fa1);
		// final Dist<double[]> gDist = new EuclideanDist(ga);

		final int nrPrototypes = 16;

		List<double[]> training = new ArrayList<double[]>();
		List<double[]> testing = new ArrayList<double[]>();

		for (int i = 0; i < samples.size(); i++)
			if( i <  samples.size()*30.0/100 )
				testing.add(samples.get(i));
			else
				training.add(samples.get(i));

		// Map<double[],Set<double[]>> map = Clustering.kMeans(samples, nrPrototypes, fDist);
		// NG
		Sorter<double[]> s;
		s = new DefaultSorter<double[]>(fDist);
		// s = new KangasSorter<double[]>(gDist, fDist, radius);
		NG ng = new NG(nrPrototypes, (double) nrPrototypes / 2, 0.01, 0.5, 0.005, samples.get(0).length, s);

		for (int t = 0; t < T_MAX * 4; t++)
			ng.train((double) t / T_MAX, samples.get(r.nextInt(samples.size())));
		Map<double[], Set<double[]>> map = NGUtils.getBmuMapping(samples, ng.getNeurons(), s);

		Map<double[], Double> hidden = new HashMap<double[], Double>();
		// min plus overlap
		for (double[] c : map.keySet()) {
			double d = Double.MAX_VALUE;
			for (double[] n : map.keySet())
				if (c != n)
					d = Math.min(d, fDist.dist(c, n)) * 1.1;
			hidden.put(c, d);
		}

		RBF rbf = new RBF(hidden, 1, fDist, 0.05);
		for (int i = 0; i < T_MAX; i++) {
			int j = r.nextInt(samples.size());
			rbf.train(samples.get(j), desired.get(j));
		}

		List<double[]> response = new ArrayList<double[]>();
		List<double[]> desiredResponse = new ArrayList<double[]>();
		for (double[] d : testing) {
			response.add(rbf.present(d));
			desiredResponse.add(desired.get(samples.indexOf(d)));
		}
		
		List<double[]> origTesting = new ArrayList<double[]>();
		for( double[] d : testing )
			origTesting.add( df.samples.get(testing.indexOf(d)));
				
		DataUtils.writeCSV("output/testing.csv", origTesting, df.names.toArray(new String[0]));
		
		List<double[]> origTraining = new ArrayList<double[]>();
		for( double[] d : training )
			origTraining.add( df.samples.get(training.indexOf(d)));
		
		DataUtils.writeCSV("output/training.csv", training, df.names.toArray(new String[0]));
		
		List<double[]> out = new ArrayList<double[]>();
		for( double[] d : testing ) {
			out.add( new double[]{ df.samples.get( testing.indexOf(d) )[0], rbf.present(d)[0] } );
		}
		
		DataUtils.writeCSV("output/rbf_output.csv", out, new String[]{"No,predicted"});
				
		log.debug("RMSE: "+Meuse.getRMSE(response, desiredResponse) );
		log.debug("R^2: "+Math.pow(Meuse.getPearson(response, desiredResponse), 2) );
	}
}
