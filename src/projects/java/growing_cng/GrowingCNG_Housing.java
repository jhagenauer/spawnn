package growing_cng;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

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

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.Point;

public class GrowingCNG_Housing {

	private static Logger log = Logger.getLogger(GrowingCNG_Housing.class);

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
		
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		List<double[]> neurons = new ArrayList<double[]>();
		for( int i = 0; i < 2; i++ ) {
			double[] d = samples.get(i);
			neurons.add( Arrays.copyOf(d, d.length) );
		}
		double ratio = 0.25;
		GrowingCNG ng = new GrowingCNG(neurons, 0.05, 0.0005, gDist, fDist, ratio, 80, 300, 0.5, 0.0005);		
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));
			ng.train( t, x );
		}
		Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), 
				new KangasSorter<double[]>(gDist, fDist, (int)Math.ceil( ng.getNeurons().size()*ratio) ) );
		log.debug("Neurons: "+ng.getNeurons().size()+", Conns: "+ng.getConections().size() );
		log.debug("fQE: "+DataUtils.getMeanQuantizationError(mapping, fDist));
		log.debug("sQE: "+DataUtils.getMeanQuantizationError(mapping, gDist));
		geoDrawNG("output/connections.png",ng.getNeurons(),ng.getConections(),ga,sdf.crs);
		Drawer.geoDrawCluster(mapping.values(), samples, geoms, "output/cluster.png", true);
	}

	public static void geoDrawNG(String fn, List<double[]> neurons, Map<Connection, Integer> conections, int[] ga, CoordinateReferenceSystem crs ) {
		GeometryFactory gf = new GeometryFactory();
		
		DefaultFeatureCollection pointFeatures = new DefaultFeatureCollection();
		{
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("neuron");
			typeBuilder.setCRS(crs);
			typeBuilder.add("the_geom", Point.class);
			
			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
			for ( double[] n : neurons ) {
				Point p = gf.createPoint( new Coordinate( n[ga[0]], n[ga[1]]));
				featureBuilder.set("the_geom", p);
				pointFeatures.add(featureBuilder.buildFeature("" + pointFeatures.size()));
			}
		}
		
		DefaultFeatureCollection lineFeatures = new DefaultFeatureCollection();
		{
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("connection");
			typeBuilder.setCRS(crs);
			typeBuilder.add("the_geom", LineString.class);
				
			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
			for (Connection c : conections.keySet() ) {
				Coordinate a = new Coordinate(c.getA()[0], c.getA()[1] );
				Coordinate b = new Coordinate(c.getB()[0], c.getB()[1] );
				LineString ls = gf.createLineString(new Coordinate[] { a, b });
				featureBuilder.set("the_geom", ls);
				lineFeatures.add(featureBuilder.buildFeature("" + lineFeatures.size()));
			}
		}

		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();
			mc.addLayer(new FeatureLayer(lineFeatures, SLD.wrapSymbolizers(sb.createLineSymbolizer(Color.BLACK, 2.0))));
			Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, Color.RED);
			mc.addLayer(new FeatureLayer(pointFeatures, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null)))));

			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);

			Rectangle imageBounds = null;

			ReferencedEnvelope mapBounds = mc.getMaxBounds();
			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 2000;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));
			System.out.println("points: "+pointFeatures.size()+", lines: "+lineFeatures.size()+", bounds: "+imageBounds);

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
