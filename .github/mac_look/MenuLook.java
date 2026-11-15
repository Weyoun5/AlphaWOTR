import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.instrument.Instrumentation;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import javax.imageio.ImageIO;
import javax.swing.AbstractButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JPopupMenu;
import javax.swing.JSlider;
import javax.swing.MenuSelectionManager;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;

/**
 * TEST ONLY (the Mac workflow): what the game client's AI menu holds and shows on this Mac.
 *
 * A java agent (-javaagent:MenuLook.jar=FOLDER through JAVA_TOOL_OPTIONS) in the REAL client, started as a person
 * starts it (Play_vs_AI.command's wotr_bridge/play.py).  It waits for the game's window and its "AI" menu, writes
 * what is there to FOLDER/look.json (the look and feel, whether the menu bar is drawn in the game's window - a bar
 * macOS draws at the top of the screen leaves the slider out -, every entry of the AI menu, the strength slider:
 * found, at which place, showing when the menu is open), opens the menu from the window's bar and keeps
 * three pictures of it (menu_paint.png: the open menu painted into an image; screen_robot.png and
 * screen_capture.png: the screen, by Java and by macOS), and then ends the client: 0 = looked, 3 = no AI menu came.
 * Nothing here knows the client's classes: the menu is found by its title, the slider by its name.
 */
public class MenuLook {
    static final String SLIDER = "ai.strength";
    static final int WAIT_S = 240;
    static File dir;
    static final StringBuilder json = new StringBuilder("{\n");

    public static void premain(String args, Instrumentation inst) {
        dir = new File(args);
        Thread t = new Thread(new Runnable() { public void run() { look(); } }, "MenuLook");
        t.setDaemon(true);
        t.start();
    }

    static String q(String s) {
        if (s == null) return "null";
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ") + "\"";
    }

    static void put(String key, String rawValue) {
        json.append(" ").append(q(key)).append(": ").append(rawValue).append(",\n");
    }

    static void finish(int code) {
        put("exit", String.valueOf(code));
        String text = json.toString();
        text = text.substring(0, text.length() - 2) + "\n}\n";
        try {
            Files.write(new File(dir, "look.json").toPath(), text.getBytes(StandardCharsets.UTF_8));
        } catch (Throwable t) {
            t.printStackTrace();
        }
        Runtime.getRuntime().halt(code);
    }

    static JMenu aiMenu() {
        Frame[] frames = Frame.getFrames();
        for (int i = 0; i < frames.length; i++) {
            if (!(frames[i] instanceof JFrame) || !frames[i].isShowing()) continue;
            JMenuBar bar = ((JFrame) frames[i]).getJMenuBar();
            if (bar == null) continue;
            for (int k = 0; k < bar.getMenuCount(); k++) {
                JMenu m = bar.getMenu(k);
                if (m != null && "AI".equals(m.getText())) return m;
            }
        }
        return null;
    }

    static JSlider slider(Component c) {
        if (c instanceof JSlider && SLIDER.equals(c.getName())) return (JSlider) c;
        if (c instanceof Container) {
            Component[] kids = ((Container) c).getComponents();
            for (int i = 0; i < kids.length; i++) {
                JSlider s = slider(kids[i]);
                if (s != null) return s;
            }
        }
        return null;
    }

    static String label(Component c) {
        if (c instanceof JLabel) return ((JLabel) c).getText();
        if (c instanceof Container) {
            Component[] kids = ((Container) c).getComponents();
            for (int i = 0; i < kids.length; i++) {
                String s = label(kids[i]);
                if (s != null) return s;
            }
        }
        return null;
    }

    static String windows() {
        StringBuilder b = new StringBuilder("[");
        Window[] all = Window.getWindows();
        for (int i = 0; i < all.length; i++) {
            Window w = all[i];
            String title = w instanceof Frame ? ((Frame) w).getTitle()
                    : w instanceof java.awt.Dialog ? ((java.awt.Dialog) w).getTitle() : "";
            b.append(i == 0 ? "" : ", ").append(q(w.getClass().getName() + " | " + title + " | showing=" + w.isShowing()
                    + " | " + w.getBounds().width + "x" + w.getBounds().height));
        }
        return b.append("]").toString();
    }

    static void pictures(String stem) {
        try {
            Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
            ImageIO.write(new Robot().createScreenCapture(new Rectangle(screen)), "png", new File(dir, stem + "_robot.png"));
        } catch (Throwable t) {
            put(stem + "_robot_error", q(String.valueOf(t)));
        }
        try {
            new ProcessBuilder("/usr/sbin/screencapture", "-x", new File(dir, stem + "_capture.png").getPath())
                    .inheritIO().start().waitFor();
        } catch (Throwable t) {
            put(stem + "_capture_error", q(String.valueOf(t)));
        }
    }

    static void look() {
        try {
            Thread.sleep(15000);                                     // the client's own start first
            put("os", q(System.getProperty("os.name") + " " + System.getProperty("os.version") + " " + System.getProperty("os.arch")));
            put("java", q(System.getProperty("java.version") + " " + System.getProperty("java.vendor")));
            JMenu menu = null;
            for (int s = 15; s < WAIT_S && menu == null; s += 2) {
                menu = aiMenu();
                if (menu == null) Thread.sleep(2000);
            }
            put("look_and_feel", q(UIManager.getLookAndFeel().getClass().getName()));
            put("swing.defaultlaf", q(System.getProperty("swing.defaultlaf")));
            put("apple.laf.useScreenMenuBar", q(System.getProperty("apple.laf.useScreenMenuBar")));
            put("windows", windows());
            if (menu == null) {
                put("ai_menu", "false");
                pictures("screen");
                finish(3);
            }
            put("ai_menu", "true");
            final JMenu ai = menu;
            Thread.sleep(3000);                                      // the menu's own build has ended
            SwingUtilities.invokeAndWait(new Runnable() {
                public void run() {
                    JMenuBar bar = (JMenuBar) ai.getParent();
                    // a bar macOS draws at the top of the screen has no height in the window, and shows menu
                    // entries only: what this agent opens below is then NOT what a person's click opens
                    put("menu_bar_in_window", String.valueOf(bar.isShowing() && bar.getHeight() > 0));
                    put("menu_bar_ui", q(bar.getUI().getClass().getName()));
                    put("menu_bar_bounds", q(String.valueOf(bar.getBounds())));
                    StringBuilder entries = new StringBuilder("[");
                    int at = -1;
                    Component[] kids = ai.getMenuComponents();
                    for (int i = 0; i < kids.length; i++) {
                        Component c = kids[i];
                        String words = c instanceof AbstractButton ? ((AbstractButton) c).getText() : label(c);
                        if (slider(c) != null) at = i;
                        entries.append(i == 0 ? "" : ", ").append(q(c.getClass().getSimpleName() + (words == null ? "" : ": " + words)
                                + (slider(c) != null ? " [the strength slider]" : "")));
                    }
                    put("entries", entries.append("]").toString());
                    put("slider_in_menu", String.valueOf(at >= 0));
                    put("slider_entry", String.valueOf(at));
                    ai.doClick();                                    // opens the menu as a click on its title does
                }
            });
            Thread.sleep(1500);
            SwingUtilities.invokeAndWait(new Runnable() {
                public void run() {
                    JPopupMenu popup = ai.getPopupMenu();
                    put("menu_open", String.valueOf(popup.isShowing()));
                    put("menu_size", q(popup.getWidth() + "x" + popup.getHeight()));
                    JSlider s = slider(popup);
                    put("slider_showing", String.valueOf(s != null && s.isShowing()));
                    if (s != null) {
                        put("slider_value", String.valueOf(s.getValue()));
                        put("slider_size", q(s.getWidth() + "x" + s.getHeight()));
                        if (s.isShowing()) put("slider_on_screen", q(String.valueOf(s.getLocationOnScreen())));
                    }
                    try {
                        if (popup.getWidth() > 0 && popup.getHeight() > 0) {
                            BufferedImage image = new BufferedImage(popup.getWidth(), popup.getHeight(), BufferedImage.TYPE_INT_RGB);
                            Graphics2D g = image.createGraphics();
                            popup.paint(g);
                            g.dispose();
                            ImageIO.write(image, "png", new File(dir, "menu_paint.png"));
                        }
                    } catch (Throwable t) {
                        put("menu_paint_error", q(String.valueOf(t)));
                    }
                }
            });
            pictures("screen");
            SwingUtilities.invokeAndWait(new Runnable() {
                public void run() { MenuSelectionManager.defaultManager().clearSelectedPath(); }
            });
            finish(0);
        } catch (Throwable t) {
            put("error", q(String.valueOf(t)));
            finish(4);
        }
    }
}
