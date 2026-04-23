import type { ExpoConfig } from "expo/config";

const buildProfile = process.env.EAS_BUILD_PROFILE ?? process.env.EXPO_PUBLIC_RELEASE_CHANNEL ?? "development";
const strictTransportRequired = buildProfile === "production";

const config: ExpoConfig = {
  name: "PitchCheck",
  slug: "pitchcheck-mobile",
  scheme: "pitchcheck",
  version: "1.0.0",
  orientation: "portrait",
  userInterfaceStyle: "dark",
  icon: "../src-tauri/icons/icon.png",
  ios: {
    supportsTablet: true,
    bundleIdentifier: process.env.EXPO_PUBLIC_IOS_BUNDLE_ID ?? "com.pitchcheck.mobile",
    buildNumber: process.env.EXPO_PUBLIC_IOS_BUILD_NUMBER ?? "1",
    infoPlist: {
      ITSAppUsesNonExemptEncryption: false,
      NSAppTransportSecurity: {
        NSAllowsArbitraryLoads: false,
      },
    },
  },
  android: {
    package: process.env.EXPO_PUBLIC_ANDROID_PACKAGE ?? "com.pitchcheck.mobile",
    versionCode: Number(process.env.EXPO_PUBLIC_ANDROID_VERSION_CODE ?? "1"),
  },
  updates: {
    fallbackToCacheTimeout: 0,
  },
  runtimeVersion: {
    policy: "appVersion",
  },
  extra: {
    eas: {
      projectId: process.env.EXPO_PUBLIC_EAS_PROJECT_ID,
    },
    buildProfile,
    strictTransportRequired,
  },
};

export default config;
