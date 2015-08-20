package llm;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.Mark;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.gui.DistanceDialog.DistMode;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.utils.SomUtils;
import spawnn.utils.Clustering;
import spawnn.utils.ColorBrewerUtil;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import cng_houseprice.OptimizeHousingLLMCNG;

import com.vividsolutions.jts.geom.Geometry;

public class LLM_Housing {

	private static Logger log = Logger.getLogger(LLM_Housing.class);

	public static void main(String[] args) {
		final DecimalFormat df = new DecimalFormat("00");
		final Random r = new Random();
		final int T_MAX = 10000;
		
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
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		for( int l = 3; l < 4; l++ ) {
		
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
		log.debug("RMSE: "+Meuse.getRMSE(response, desired)+", R2: "+Math.pow(Meuse.getPearson(response, desired), 2));
		
		/*for( double[] d : ng.getNeurons() ) {
			log.debug("Prt: "+Arrays.toString(d) );
			log.debug("Mat: "+Arrays.toString(ng.matrix.get(d)[0]));
			log.debug("Out: "+Arrays.toString(ng.output.get(d)));
		}*/
		
		//Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, llm.getNeurons(), sorter ).values();
		Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter);
		Drawer.geoDrawCluster(mapping.values(), samples, geoms, "output/cluster_"+df.format(l)+".png", true);
		try {
			SomUtils.printDMatrix(grid, fDist, ColorMode.Blues, new FileOutputStream("output/dmatrix_"+df.format(l)+".png"));
						
			Dist<double[]> eDist = new EuclideanDist();
			for( GridPos p : grid.getPositions() )
				grid.setPrototypeAt(p, llm.output.get(p) );
			SomUtils.printDMatrix(grid, eDist, ColorMode.Blues, new FileOutputStream("output/dmatrix_output_"+df.format(l)+".png"));
			
			for( GridPos p : grid.getPositions() )
				grid.setPrototypeAt(p, llm.matrix.get(p)[0] );
			SomUtils.printDMatrix(grid, eDist, ColorMode.Blues, new FileOutputStream("output/dmatrix_matrix_"+df.format(l)+".png"));
			
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
	
	public static void geoDrawValues(List<Geometry> geoms, List<Double> values, CoordinateReferenceSystem crs, ColorMode cm, String fn) {
		SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
		typeBuilder.setName("data");
		typeBuilder.setCRS(crs);
		typeBuilder.add("the_geom", geoms.get(0).getClass());

		SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
		Map<Geometry, Double> m = new HashMap<Geometry, Double>();
		for (int i = 0; i < geoms.size(); i++)
			m.put(geoms.get(i), values.get(i));

		Map<Geometry, Color> colMap = ColorBrewerUtil.valuesToColors(m, cm);
		Set<Color> cols = new HashSet<Color>(colMap.values());

		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();

			ReferencedEnvelope mapBounds = mc.getMaxBounds();
			for (Color c : cols) {
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				for (Geometry g : colMap.keySet()) {
					if (!c.equals(colMap.get(g)))
						continue;
					featureBuilder.set("the_geom", g);
					features.add(featureBuilder.buildFeature("" + features.size()));
				}
				Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, c);
				FeatureLayer fl = new FeatureLayer(features, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null))));
				mc.addLayer(fl);
				mapBounds.expandToInclude(fl.getBounds());
			}

			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);

			Rectangle imageBounds = null;
			
			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 2000;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));

			BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D gr = image.createGraphics();
			renderer.paint(gr, imageBounds, mapBounds);

			ImageIO.write(image, "png", new FileOutputStream(fn));
			image.flush();
			mc.dispose();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
