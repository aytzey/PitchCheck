import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import DesktopWorkbench from "@/components/DesktopWorkbench";

describe("DesktopWorkbench", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/");
  });

  it("opens the settings panel from the URL hash with OpenRouter defaults", () => {
    window.history.replaceState(null, "", "/#settings");

    render(<DesktopWorkbench />);

    expect(screen.getByText("Machine env")).toBeDefined();
    expect(screen.getByLabelText(/OpenRouter API key/)).toBeDefined();
    expect(screen.getByLabelText<HTMLInputElement>(/OpenRouter model/).value).toBe(
      "anthropic/claude-sonnet-4.6",
    );
  });

  it("updates the URL hash when changing tabs", () => {
    render(<DesktopWorkbench />);

    fireEvent.click(screen.getByRole("button", { name: "Runtime" }));

    expect(window.location.hash).toBe("#runtime");
    expect(screen.getByText("03 / Runtime selection")).toBeDefined();
  });

  it("exposes refine and variant re-rank controls in the workspace", async () => {
    render(<DesktopWorkbench />);

    expect(screen.getByText("Variant re-rank")).toBeDefined();
    expect(screen.getByRole("button", { name: /Refine & re-score/ })).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /Refine & re-score/ }));

    await waitFor(() => expect(screen.getByText("After . refined")).toBeDefined());
    expect(screen.getByRole("button", { name: "Accept & rescore" })).toBeDefined();
  });
});
