package llm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomUtils;
import spawnn.utils.Clustering;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class LLMSOM_Housing {

	private static Logger log = Logger.getLogger(LLMSOM_Housing.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 100000;
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[]{6,7},new int[]{}, true);

		final List<double[]> samples = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		final List<double[]> desired = new ArrayList<double[]>();
		
		List<String> vars = new ArrayList<String>();
		vars.add("xco");
		vars.add("yco");
		vars.add("lnarea_tot");
		vars.add("lnarea_plo");
		vars.add("attic_dum");
		vars.add("cellar_dum");
		vars.add("cond_house_3");
		vars.add("heat_3");
		vars.add("bath_3");
		vars.add("garage_3");
		vars.add("terr_dum");
		vars.add("age_num");
		vars.add("time_index");
		vars.add("zsp_alq_09");
		vars.add("gem_kauf_i");
		vars.add("gem_abi");
		vars.add("gem_alter_");
		vars.add("ln_gem_dic");

		for( double[] d : sdf.samples ) {
			if( d[sdf.names.indexOf("time_index")] < 6 )
				continue;
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			for( int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))];
			samples.add(nd);
			desired.add( new double[]{ d[sdf.names.indexOf("lnp")] } );
			geoms.add( sdf.geoms.get(idx) );
		}
		
		final int[] fa = new int[]{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
		final int[] ga = new int[]{0, 1};
		DataUtils.zScoreColumns(samples,fa);
		
		final Dist<double[]> gDist = new EuclideanDist(new int[]{0,1});
		final Dist<double[]> fDist = new EuclideanDist(fa);

		Sorter<double[]> sorter = new DefaultSorter<double[]>(fDist);
		sorter = new ErrorSorter(samples, desired);
		LLMNG ng = new LLMNG(9, 5, 0.01, 0.5, 0.005, 
				5, 0.01, 0.5, 0.005, 
				sorter, fa, samples.get(0).length, 1);
		((ErrorSorter)sorter).setLLMNG(ng);
								
		for (int t = 0; t < 100000; t++) {
			int j = r.nextInt(samples.size());
			ng.train( (double)t/T_MAX, samples.get(j), desired.get(j) );
		}
		
		List<double[]> response = new ArrayList<double[]>();
		for (double[] x : samples)
			response.add(ng.present(x));
		log.debug("RMSE: "+Meuse.getRMSE(response, desired)+", R2: "+Math.pow(Meuse.getPearson(response, desired), 2));
		
		Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter );
		Drawer.geoDrawCluster(mapping.values(), samples, geoms, "output/cluster.png", true);
						
		{
			List<double[]> res = new ArrayList<double[]>();
			List<double[]> des = new ArrayList<double[]>();
			for (Set<double[]> s : mapping.values() ) {
				List<Integer> toIgnore = new ArrayList<Integer>();
				for( int i : ga )
					toIgnore.add(i);
				
				for( int i = 0; i < s.iterator().next().length; i++ ) {
					DescriptiveStatistics ds = new DescriptiveStatistics();
					for( double[] d : s )
						ds.addValue( d[i] );
					if( ds.getStandardDeviation() < 0.00001 )
						toIgnore.add(i);
				}
								
				double[] y = new double[s.size()];
				double[][] x = new double[s.size()][];
				int l = 0;
				for (double[] d : s) {
					int idx = samples.indexOf(d);
					y[l] = desired.get(idx)[0];
					x[l] = new double[d.length-toIgnore.size()];
					int j = 0;
					for( int i = 0; i < d.length; i++) {
						if( toIgnore.contains(i) )
							continue;
						x[l][j++] = d[i];
					}
					l++;
				}

				OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
				ols.setNoIntercept(false);
				ols.newSampleData(y, x);
				double[] beta = ols.estimateRegressionParameters();

				for (int i = 0; i < x.length; i++) {
					double[] xi  = Arrays.copyOf(x[i], x[i].length);
					double ps = beta[0]; // intercept at beta[0]
					for (int j = 1; j < beta.length; j++)
						ps += beta[j] * xi[j - 1];

					res.add(new double[] { ps });
					des.add( new double[]{y[i]} );
				}
				
			}
			log.debug("RMSE: "+Meuse.getRMSE(res, des)+", R2: "+Math.pow(Meuse.getPearson(res, des), 2));
		}
	}
}
