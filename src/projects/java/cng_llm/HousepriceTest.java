package cng_llm;

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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import llm.ErrorSorter;
import llm.LLMNG;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.geotools.factory.CommonFactoryFinder;
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
import org.opengis.filter.FilterFactory2;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ColorBrewerUtil;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GraphClustering;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.Point;

public class HousepriceTest {

	private static Logger log = Logger.getLogger(HousepriceTest.class);

	public static void main(String[] args) {
		final DecimalFormat df = new DecimalFormat("00");
		final Random r = new Random();
		final int T_MAX = 40000;

		final List<double[]> samples = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		final List<double[]> desired = new ArrayList<double[]>();
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);
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

		final int da = sdf.names.indexOf("lnp");
		for (double[] d : sdf.samples) {
			if (d[sdf.names.indexOf("time_index")] < 6)
				continue;
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))]; 
					
			samples.add(nd);
			desired.add(new double[]{d[da]});
			geoms.add(sdf.geoms.get(idx));
		}

		final int[] fa = new int[vars.size()-2]; // omit geo-vars
		for( int i = 0; i < fa.length; i++ )
			fa[i] = i+2;
		final int[] ga = new int[] { 0, 1 };

		// ------------------------------------------------------------------------
		
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(samples, da); // should not be necessary

		{ // just write it to csv inkl. csv for external use
			List<double[]> l = new ArrayList<double[]>();
			for( double[] d : samples ) {
				double[] nd = Arrays.copyOf(d, d.length+1);
				nd[nd.length-1] = desired.get(samples.indexOf(d))[0];
				l.add(nd);
			}
			List<String> nv = new ArrayList<String>(vars);
			nv.add("lnp");
			DataUtils.writeCSV("output/houseprice.csv", l, nv.toArray(new String[] {}));
		}
			
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		int nrNeurons = 30;
		int nrCluster = 12;
		
		for (int run = 0; run < 1; run++)
			for (int l = 1; l <= nrNeurons/2; l++) {

				List<double[]> neurons = new ArrayList<double[]>();
				for (int i = 0; i < nrNeurons; i++) {
					double[] d = samples.get(r.nextInt(samples.size()));
					neurons.add(Arrays.copyOf(d, d.length));
				}

				ErrorSorter errorSorter = new ErrorSorter(samples, desired);
				DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
				Sorter<double[]> sorter = new KangasSorter<>(gSorter, errorSorter, l);
				
				DecayFunction nbRate = new PowerDecay(nrNeurons/3, 0.1);
				DecayFunction lrRate1 = new PowerDecay(0.5, 0.001);
				DecayFunction lrRate2 = new PowerDecay(0.1, 0.001);
				LLMNG ng = new LLMNG(neurons, 
						nbRate, lrRate1, 
						nbRate, lrRate2, 
						sorter, fa, 1, true );
				errorSorter.setLLMNG(ng);

				for (int t = 0; t < T_MAX; t++) {
					int j = r.nextInt(samples.size());
					ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
				}
				
				List<double[]> response = new ArrayList<double[]>();
				for (double[] x : samples)
					response.add(ng.present(x));

				Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, neurons, sorter);
				
				log.debug(l+", RMSE: " + Meuse.getRMSE(response, desired) + ", R2: " + Math.pow(Meuse.getPearson(response, desired), 2));
				DataUtils.writeCSV("output/cng_proto", neurons, vars.toArray(new String[]{}));
				
				// coefs ----------------------------------------------------------
				Map<double[],double[]> geoCoefs = new HashMap<double[],double[]>();
				for( double[] n : ng.getNeurons() ) {
					double[] cg = new double[2+fa.length];
					cg[0] = n[ga[0]];
					cg[1] = n[ga[1]];
					for( int i = 0; i < fa.length; i++ )
						cg[2+i] = ng.matrix.get(n)[0][i];
					geoCoefs.put(n, cg);
				}
				DataUtils.writeCSV("output/cng_coefs_"+l+".csv", geoCoefs.values(), vars.toArray(new String[]{}));
												
				// output/intercept ----------------------------------------------------
				Map<double[],double[]> geoOut = new HashMap<double[],double[]>();
				for( double[] n : ng.getNeurons() ) {
					double[] cg = new double[ga.length+1];
					cg[0] = n[ga[0]];
					cg[1] = n[ga[1]];
					cg[2] = ng.output.get(n)[0];
					geoOut.put(n, cg);
				}
				DataUtils.writeCSV("output/cng_output_"+l+".csv", geoOut.values(), new String[]{"xco","yco","output"});
				
				// write cluster to csv
				int i = 0;
				List<double[]> ns = new ArrayList<double[]>();
				for( double[] n : mapping.keySet() ) {
					for( double[] d : mapping.get(n) ) {
						double[] nd = Arrays.copyOf(d, d.length+1);
						nd[nd.length-1] = i;
						ns.add(nd);
					}
					i++;
				}
				List<String> nVars = new ArrayList<String>(vars);
				nVars.add("neuron");
				DataUtils.writeCSV("output/cng_cluster_"+l+".csv", ns, nVars.toArray( new String[]{} ) );	
								
				// CHL 
				Map<Connection, Integer> conns = new HashMap<Connection, Integer>();
				for (double[] x : samples) {
					sorter.sort(x, ng.getNeurons());
					List<double[]> bmuList = ng.getNeurons();

					Connection c = new Connection(bmuList.get(0), bmuList.get(1));
					if (!conns.containsKey(c))
						conns.put(c, 1);
					else
						conns.put(c, conns.get(c) + 1);
				}
				geoDrawNG("output/cng_graph_"+df.format(l)+".png", mapping, conns, ga, null);
				
				if( nrCluster < nrNeurons ) {
					Map<double[],Map<double[],Double>> graph = new HashMap<double[],Map<double[],Double>>();
					for( Connection c : conns.keySet() ) {
						double[] a = c.getA();
						double[] b = c.getB();
						double v = conns.get(c);
						if( !graph.containsKey(a) )
							graph.put(a, new HashMap<double[],Double>() );
						graph.get(a).put(b, v);
						if( !graph.containsKey(b))
							graph.put(b, new HashMap<double[],Double>() );
						graph.get(b).put(a, v);
					}
					Map<double[],Integer> clust = GraphClustering.greedyOptModularity(graph, 10, nrCluster);
					Map<Integer,Set<double[]>> cluster = new HashMap<Integer,Set<double[]>>();
					for( double[] d : clust.keySet() ) {
						int c = clust.get(d);
						if( !cluster.containsKey(c) )
							cluster.put(c, new HashSet<double[]>());
						cluster.get(c).add(d);
					}
					Drawer.geoDrawCluster(cluster.values(), samples, geoms, "output/cng_graph_cluster_"+l+".png", true);
				}
				
				{ 	// d-matrix
					Map<double[],Double> vMap = new HashMap<double[],Double>();
					for( double[] n : ng.getNeurons() ) {
						DescriptiveStatistics ds = new DescriptiveStatistics();
						for( double[] nb : Connection.getNeighbors(conns.keySet(), n, 1))
							ds.addValue( fDist.dist(n, nb));
						vMap.put( n, ds.getMean() );
					}
					SpatialHetBacaoTest.geoDrawNG("output/cng_proto_dmatrix_"+df.format(l)+".png", vMap, conns.keySet(), ga, samples, 5);
				}
				
				List<Connection> coefConns = new ArrayList<Connection>();
				for( Connection c : conns.keySet() )
					coefConns.add( new Connection( geoCoefs.get(c.getA()), geoCoefs.get(c.getB() )) );
				
				{ 	// d-matrix coef
					Map<double[],Double> vMap = new HashMap<double[],Double>();
					for( double[] d : ng.getNeurons() ) {
						DescriptiveStatistics ds = new DescriptiveStatistics();
						double[] coef = geoCoefs.get(d);
						for( double[] nb : Connection.getNeighbors(coefConns, coef, 1))
							ds.addValue( fDist.dist(coef, nb));
						vMap.put( coef, ds.getMean() );
					}
					SpatialHetBacaoTest.geoDrawNG("output/cng_coef_dmatrix_"+df.format(l)+".png", vMap, coefConns, ga, samples, 5);
				}
				
				
				List<Connection> outConns = new ArrayList<Connection>();
				for( Connection c : conns.keySet() )
					outConns.add( new Connection( geoOut.get(c.getA()), geoOut.get(c.getB() )) );
							
				// components
				{
					Map<double[],Double> vMap = new HashMap<double[],Double>();
					for( double[] d : ng.getNeurons() ) {
						double[] out = geoOut.get(d);
						vMap.put(out, out[2]);
					}
					SpatialHetBacaoTest.geoDrawNG("output/ng_output_"+df.format(l)+"_"+df.format(run)+".png", vMap, outConns, ga, samples, 5);
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
	
	public static void geoDrawNG(String fn, Map<double[], Set<double[]>> mapping, Map<Connection, Integer> conns, int[] ga, CoordinateReferenceSystem crs) {
		FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2();
		GeometryFactory gf = new GeometryFactory();

		List<double[]> neurons = new ArrayList<double[]>(mapping.keySet());
		Map<double[], Double> values = new HashMap<double[], Double>();
		for (double[] d : neurons)
			values.put(d, (double) neurons.indexOf(d));
		Map<double[], Color> col = ColorBrewerUtil.valuesToColors(values, ColorMode.Spectral);

		StyleBuilder sb = new StyleBuilder();
		MapContent mc = new MapContent();
		ReferencedEnvelope mapBounds = mc.getMaxBounds();
		for (double[] n : neurons) {
			if( mapping.get(n).isEmpty() )
				continue;
			// samples
			DefaultFeatureCollection sampleFeatures = new DefaultFeatureCollection();
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("sample");
				typeBuilder.setCRS(crs);
				typeBuilder.add("the_geom", Point.class);

				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				for (double[] d : mapping.get(n)) {
					Point p = gf.createPoint(new Coordinate(d[ga[0]], d[ga[1]]));
					featureBuilder.set("the_geom", p);
					sampleFeatures.add(featureBuilder.buildFeature("" + sampleFeatures.size()));
				}
			}
			Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, col.get(n));
			FeatureLayer fl = new FeatureLayer(sampleFeatures, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null)))); 
			mc.addLayer(fl);
			mapBounds.expandToInclude(fl.getBounds());
		}
		
		DefaultFeatureCollection lineFeatures = new DefaultFeatureCollection();
		{
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("connection");
			typeBuilder.setCRS(crs);
			typeBuilder.add("weight",Double.class);
			typeBuilder.add("the_geom", LineString.class);
			
			double min = Collections.min( conns.values() );
			double max = Collections.max( conns.values() );

			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
			for (Connection c : conns.keySet()) {
				Coordinate a = new Coordinate(c.getA()[ga[0]], c.getA()[ga[1]]);
				Coordinate b = new Coordinate(c.getB()[ga[0]], c.getB()[ga[1]]);
				LineString ls = gf.createLineString(new Coordinate[] { a, b });
				featureBuilder.set("weight", 1+10*(conns.get(c)-min)/(max-min) );
				featureBuilder.set("the_geom", ls);
				lineFeatures.add(featureBuilder.buildFeature("" + lineFeatures.size()));
			}
		}
		mc.addLayer(new FeatureLayer(lineFeatures, SLD.wrapSymbolizers(sb.createLineSymbolizer(sb.createStroke(ff.literal("BLACK"),ff.property("weight"))))));
		
		for( double[] n : neurons ) {
			// neuron
			DefaultFeatureCollection neuronFeatures = new DefaultFeatureCollection();
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("neuron");
				typeBuilder.setCRS(crs);
				typeBuilder.add("the_geom", Point.class);

				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				Point p = gf.createPoint(new Coordinate(n[ga[0]], n[ga[1]]));
				featureBuilder.set("the_geom", p);
				neuronFeatures.add(featureBuilder.buildFeature("" + neuronFeatures.size()));
			}
			Mark mark2 = sb.createMark(StyleBuilder.MARK_CIRCLE, col.get(n), Color.BLACK, 2.0 );
			mc.addLayer(new FeatureLayer(neuronFeatures, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark2, null)))));
		}
		
		GTRenderer renderer = new StreamingRenderer();
		renderer.setMapContent(mc);

		double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
		int imageWidth = 2000;
		Rectangle imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));

		BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
		Graphics2D gr = image.createGraphics();
		renderer.paint(gr, imageBounds, mapBounds);

		try {
			ImageIO.write(image, "png", new FileOutputStream(fn));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		image.flush();
		mc.dispose();

	}
}
