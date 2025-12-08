import streamlit as st
import re

st.title("WAPA Homepage Slider Replacement Tool")
st.write("""
Paste the full homepage HTML below, enter new slider images/links, and this tool will **replace only the carousel HTML** with updated code.
YM-compatible arrows are used automatically (your custom arrow icons removed).
""")

# --- Full HTML input from staff ---
full_html_input = st.text_area(
    "Paste FULL homepage HTML here:",
    height=400,
    key="full_html",
    placeholder="Paste the entire WAPA homepage HTML..."
)

# --- Number of slides ---
num_slides = st.number_input("How many slides?", min_value=1, max_value=5, value=3)

slides = []
for i in range(num_slides):
    st.subheader(f"Slide {i+1}")
    img = st.text_input(f"Image URL for Slide {i+1}", key=f"img_{i}")
    link = st.text_input(f"Optional click-through URL for Slide {i+1}", key=f"link_{i}")
    alt = st.text_input(f"Alt text for Slide {i+1}", value=f"Slide {i+1}", key=f"alt_{i}")
    slides.append({"img": img, "link": link, "alt": alt})


def build_slider(slides):
    """Generate updated YM-compatible slider (Option 2: no custom arrow icons)."""

    html = []
    html.append('<div id="myCarousel" class="carousel slide carousel-fade" data-ride="carousel" data-interval="5000">')

    # INDICATORS
    html.append('  <ol class="carousel-indicators">')
    for i in range(len(slides)):
        active = ' class="active"' if i == 0 else ""
        html.append(f'    <li data-target="#myCarousel" data-slide-to="{i}"{active}></li>')
    html.append('  </ol>')

    # SLIDES
    html.append('  <div class="carousel-inner" role="listbox">')
    for i, slide in enumerate(slides):
        active = " active" if i == 0 else ""
        html.append(f'    <div class="item{active}" style="text-align:center;">')

        # Optional link wrapper
        if slide["link"]:
            html.append(f'      <a href="{slide["link"]}" target="_blank">')

        html.append(f'        <img alt="{slide["alt"]}" src="{slide["img"]}" />')

        if slide["link"]:
            html.append("      </a>")

        html.append("    </div>")
    html.append("  </div>")

    # CONTROLS — Option 2 (NO glyphicon arrows, let YM inject arrows)
    html.append('  <a class="left carousel-control" href="#myCarousel" role="button" data-slide="prev">')
    html.append('    <span class="sr-only">Previous</span>')
    html.append("  </a>")

    html.append('  <a class="right carousel-control" href="#myCarousel" role="button" data-slide="next">')
    html.append('    <span class="sr-only">Next</span>')
    html.append("  </a>")

    html.append("</div>")

    return "\n".join(html)


# ===============================
# Generate updated full-page HTML
# ===============================
if st.button("Generate Updated Full Page HTML"):

    if not full_html_input.strip():
        st.error("Please paste the full homepage HTML first.")
        st.stop()

    new_slider = build_slider(slides)

    # Regex pattern to find the entire existing slider block
    pattern = re.compile(
        r'<div id="myCarousel".*?</div>\s*</div>|<div id="myCarousel".*?</div>',
        re.DOTALL
    )

    if not pattern.search(full_html_input):
        st.error("Could not find an existing <div id=\"myCarousel\"> slider block to replace.")
        st.stop()

    updated_html = re.sub(pattern, new_slider, full_html_input, count=1)

    st.success("Updated full-page HTML generated!")
    st.code(updated_html, language="html")
