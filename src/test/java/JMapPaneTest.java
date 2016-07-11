import java.util.Random;

import javax.swing.JFrame;

import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.map.MapViewport;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.geotools.styling.Symbolizer;
import org.geotools.swing.JMapPane;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Point;

public class JMapPaneTest extends JFrame {
	
	public JMapPane mp;
	
	public JMapPaneTest() {
		mp = new JMapPane();
		
		GTRenderer renderer = new StreamingRenderer();
		mp.setRenderer(renderer);
				
		MapContent mc = new MapContent();
		mp.setMapContent(mc);
		add(mp);
		
		setSize(400,400);
		setVisible(true);
	}
	
	public void updateMP() {
		GeometryFactory gf = new GeometryFactory();
		SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
		typeBuilder.setName("Points");
		typeBuilder.add("the_geom",Point.class);
		
		SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
		Random r = new Random();		
		DefaultFeatureCollection fc = new DefaultFeatureCollection();
		for( int i = 0; i < 1000; i++ ) {
			Point p = gf.createPoint(new Coordinate(r.nextDouble()*100,r.nextDouble()*100));
			featureBuilder.set("the_geom", p);
			fc.add( featureBuilder.buildFeature(""+fc.size()));
		}
		
		StyleBuilder sb = new StyleBuilder();
		Symbolizer sym = sb.createPointSymbolizer();

		//MapContent mc = new MapContent();
		MapContent mc = mp.getMapContent();
		mc.layers().clear();
		
		mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sym)));
		mc.setViewport( new MapViewport(fc.getBounds()));
		
		//mp.getMapContent().dispose();
		//mp.setMapContent(mc);
	}
	
	public static void main(String[] args) {
		JMapPaneTest t = new JMapPaneTest();
		for( int i = 0; i < 1000; i++)
			t.updateMP();
		System.exit(1);
	}
}
