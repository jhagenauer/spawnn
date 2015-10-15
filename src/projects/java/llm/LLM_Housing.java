package llm;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.Grid2DHexToroid;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class LLM_Housing {

	private static Logger log = Logger.getLogger(LLM_Housing.class);

	public static void main(String[] args) {
		final DecimalFormat df = new DecimalFormat("00");
		final Random r = new Random();
		final int T_MAX = 100000;
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		final List<double[]> desired = new ArrayList<double[]>();
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[]{6,7},new int[]{}, true);
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
		
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("/home/julian/workspace/toolsAndTests/output/varimp.csv"), new int[]{0,1}, new int[]{}, true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		final List<double[]> desired = new ArrayList<double[]>();
		
		for( double[] d : samples )
			desired.add( new double[]{ d[3] } );
		
		final int[] fa = new int[]{4,5};
		final int[] ga = new int[]{2};*/
		
		// ------------------------------------------------------------------------
		
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.writeCSV("output/samples.csv", samples, vars.toArray( new String[]{}) );
		
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		for( int run = 0; run < 10; run++ )
		for( int l = 13; l <= 13; l++ ) {
		
		/*List<double[]> neurons = new ArrayList<double[]>();
		for( int i = 0; i < 10; i++ ) {
			double[] d = samples.get(r.nextInt(samples.size()));
			neurons.add( Arrays.copyOf(d, d.length));
		}

		ErrorSorter errorSorter = new ErrorSorter(samples, desired);
		Sorter<double[]> sorter = new KangasSorter<>( new DefaultSorter<>( gDist ), errorSorter, 2);
		LLMNG ng = new LLMNG(neurons, neurons.size(), 0.1, 0.5, 0.005, neurons.size(), 0.1, 0.1, 0.005, sorter, fa, 1);
		errorSorter.setLLMNG(ng);*/
		
		Grid2DHex<double[]> grid = new Grid2DHex<>(12, 8);
		//log.debug(grid.getMaxDist()); System.exit(1);
		SomUtils.initRandom(grid, samples);
		ErrorBmuGetter errorBmuGetter = new ErrorBmuGetter(samples, desired);
		BmuGetter<double[]> bmuGetter = new KangasBmuGetter<>(new DefaultBmuGetter<>(gDist), errorBmuGetter, l);
		LLMSOM llm = new LLMSOM(
				new GaussKernel( new LinearDecay(10, 0.1)), new LinearDecay(0.5, 0.005), grid, bmuGetter, 
				new GaussKernel( new LinearDecay(10, 0.1)), new LinearDecay(0.1, 0.005), fa, 1);
		errorBmuGetter.setLLMSOM(llm);
		
		for (int t = 0; t < T_MAX; t++) {
			int j = r.nextInt(samples.size());
			llm.train( (double)t/T_MAX, samples.get(j), desired.get(j) );
		}
		
		List<double[]> response = new ArrayList<double[]>();
		for (double[] x : samples)
			response.add(llm.present(x));
		// log.debug("RMSE: "+Meuse.getRMSE(response, desired)+", R2: "+Math.pow(Meuse.getPearson(response, desired), 2));
		
		/*for( double[] d : ng.getNeurons() ) {
			log.debug("Prt: "+Arrays.toString(d) );
			log.debug("Mat: "+Arrays.toString(ng.matrix.get(d)[0]));
			log.debug("Out: "+Arrays.toString(ng.output.get(d)));
		}*/
		
		//Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, llm.getNeurons(), sorter ).values();
		Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter);
		Drawer.geoDrawCluster(mapping.values(), samples, geoms, "output/cluster_"+df.format(l)+"_"+df.format(run)+".png", true);
		Map<GridPos,double[]> om = new HashMap<GridPos,double[]>(grid.getGridMap());
				
		//save mapping
		List<double[]> m = new ArrayList<double[]>();
		for( GridPos p : mapping.keySet() ) 
			for( double[] d : mapping.get(p) ) 
				m.add( new double[]{p.getPosVector()[0], p.getPosVector()[1], samples.indexOf(d)});
		DataUtils.writeCSV("output/mapping_"+df.format(l)+"_"+df.format(run)+".csv", m, new String[]{"x","y","idx"});
				
		try {
			SomUtils.printDMatrix(grid, fDist, ColorMode.Blues, new FileOutputStream("output/dmatrix_"+df.format(l)+"_"+df.format(run)+".png"));
			SomUtils.saveGrid(grid, new FileOutputStream("output/orig_grid_"+df.format(l)+"_"+df.format(run)+".xml"));
						
			Dist<double[]> eDist = new EuclideanDist();
			for( GridPos p : grid.getPositions() )
				grid.setPrototypeAt(p, llm.output.get(p) );
			SomUtils.printDMatrix(grid, eDist, ColorMode.Blues, new FileOutputStream("output/dmatrix_output_"+df.format(l)+"_"+df.format(run)+".png"));
			//SomUtils.saveGrid(grid, new FileOutputStream("output/output_grid.xml"));
			
			for( GridPos p : grid.getPositions() )
				grid.setPrototypeAt(p, llm.matrix.get(p)[0] );
			SomUtils.printDMatrix(grid, eDist, ColorMode.Blues, new FileOutputStream("output/dmatrix_matrix_"+df.format(l)+"_"+df.format(run)+".png"));
			
			for( GridPos p : grid.getPositions() ) {
				double[] d = new double[fa.length+2];
				d[0] = om.get(p)[0];
				d[1] = om.get(p)[1];
				for( int i = 0; i < fa.length; i++ )
					d[i+2] = llm.matrix.get(p)[0][i];
				grid.setPrototypeAt(p, d );
			}
			
			SomUtils.saveGrid(grid, new FileOutputStream("output/matrix_grid_"+df.format(l)+"_"+df.format(run)+".xml"));
			
			/*Map<GridPos,Double> vMap = new HashMap<GridPos,Double>();
			for (GridPos p : grid.getPositions()) {
				double[] v = grid.getPrototypeAt(p);
				DescriptiveStatistics ds = new DescriptiveStatistics();
				for (GridPos np : grid.getNeighbours(p)) 
					ds.addValue(eDist.dist(v, grid.getPrototypeAt(np)));
				vMap.put(p, ds.getMean());
			}
			List<Double> values = new ArrayList<Double>();
			for( double[] d : samples )
				for( GridPos p : mapping.keySet() )
					if( mapping.get(p).contains(d))
						values.add(vMap.get(p));
				
			OptimizeHousingLLMCNG.geoDrawValues(geoms, values, sdf.crs, "output/map_matrix_"+df.format(l)+".png");
			
			Map<double[],Set<double[]>> cm = new HashMap<double[],Set<double[]>>();
			for( GridPos p : grid.getPositions() ) {
				double[] d = grid.getPrototypeAt(p);
				Set<double[]> s = new HashSet<double[]>();
				for( GridPos nb : grid.getNeighbours(p) )
					s.add(grid.getPrototypeAt(nb));
				cm.put(d, s);
			}
			Map<Set<double[]>,TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, eDist, Clustering.HierarchicalClusteringType.ward);
			List<Set<double[]>> clust = Clustering.cutTree(tree, 9);
						
			Grid2DHex<double[]> clustGrid = new Grid2DHex<>(12, 8);
			for( int i = 0; i < clust.size(); i++ ) 
				for( double[] d : clust.get(i) )
					clustGrid.setPrototypeAt( grid.getPositionOf(d), new double[]{i} );					
			SomUtils.printComponentPlane(clustGrid, 0, ColorMode.Spectral, new FileOutputStream("output/clust_matrix_"+df.format(l)+".png"));
			
			List<Double> nValues = new ArrayList<Double>();
			for( double[] d : samples )
				for( GridPos p : mapping.keySet() )
					if( mapping.get(p).contains(d))
						nValues.add( clustGrid.getPrototypeAt(p)[0] );
			
			geoDrawValues(geoms, nValues, sdf.crs, ColorMode.Spectral, "output/clust_matrix_map_"+df.format(l)+".png");*/
						
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	}
	}
}
